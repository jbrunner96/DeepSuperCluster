from collections.abc import Iterable
import numpy as np
import awkward as ak
import numba


@numba.njit
def get_matching_pairs_indices(idx_1, idx_2, builder, builder2):
    for ev_q, ev_j in zip(idx_1, idx_2):
        builder.begin_list()
        builder2.begin_list()
        q_done = []
        j_done = []
        for i, (q, j) in enumerate(zip(ev_q, ev_j)):
            if q not in q_done:
                if j not in j_done:
                    builder.append(i)
                    q_done.append(q)
                    j_done.append(j)
                else:
                    builder2.append(i)
        builder.end_list()
        builder2.end_list()
    return builder, builder2


# This function takes as arguments the indices of two collections of objects that have been
# previously matched. The idx_matched_obj2 indices are supposed to be ordered but they can have missing elements.
# The idx_matched_obj indices are matched to the obj2 ones and no order is required on them.
# The function return an array of the dimension of the maxdim_obj2 (akward dimensions) with the indices of idx_matched_obk
# matched to the elements in idx_matched_obj2. None values are included where
# no match has been found.
@numba.njit
def get_matching_objects_indices_padnone(
    idx_matched_obj, idx_matched_obj2, maxdim_obj2, deltaR, builder, builder2, builder3
):
    for ev1_match, ev2_match, nobj2, dr in zip(
        idx_matched_obj, idx_matched_obj2, maxdim_obj2, deltaR
    ):
        # print(ev1_match, ev2_match)
        builder.begin_list()
        builder2.begin_list()
        builder3.begin_list()
        row1_length = len(ev1_match)
        missed = 0
        for i in range(nobj2):
            # looping on the max dimension of collection 2 and checking if the current index i
            # is matched, e.g is part of ev2_match vector.
            if i in ev2_match:
                # if this index is matched, then take the ev1_match and deltaR results
                # print(i, row1_length)
                builder2.append(i)
                if i - missed < row1_length:
                    builder.append(ev1_match[i - missed])
                    builder3.append(dr[i - missed])
            else:
                # If it is missing a None is added and the missed  is incremented
                # so that the next matched one will get the correct element assigned.
                builder.append(None)
                builder2.append(None)
                builder3.append(None)
                missed += 1
        builder.end_list()
        builder2.end_list()
        builder3.end_list()
    return builder, builder2, builder3


def metric_pt(obj, obj2):
    return abs(obj.pt - obj2.pt)


def get_unique_match(obj1, obj2, deltaRmax=0.2):
    a, b = ak.unzip(
       ak.cartesian([obj1, obj2], axis=1, nested=True)
        )
    deltaR = ak.flatten(a.deltaR(b), axis=2)
    # Keeping only the pairs with a deltaR min
    maskDR = deltaR < deltaRmax

    # Get the indexing to sort the pairs sorted by deltaR without any cut
    idx_pairs_sorted = ak.argsort(deltaR, axis=1)
    pairs = ak.argcartesian([obj1, obj2])
    # Sort all the collection over pairs by deltaR
    pairs_sorted = pairs[idx_pairs_sorted]
    deltaR_sorted = deltaR[idx_pairs_sorted]
    maskDR_sorted = maskDR[idx_pairs_sorted]
    idx_obj, idx_obj2 = ak.unzip(pairs_sorted)

    # Now get only the matching indices by looping over the pairs in order of deltaR.
    # The result contains the list of pairs that are considered valid
    _idx_matched_pairs, _idx_missed_pairs = get_matching_pairs_indices(
        ak.without_parameters(idx_obj, behavior={}),
        ak.without_parameters(idx_obj2, behavior={}),
        ak.ArrayBuilder(),
        ak.ArrayBuilder(),
    )

    idx_matched_pairs = _idx_matched_pairs.snapshot()
    # The indices related to the invalid jet matches are skipped
    # idx_missed_pairs  = _idx_missed_pairs.snapshot()
    # Now let's get get the indices of the objects corresponding to the valid pairs
    idx_matched_obj = idx_obj[idx_matched_pairs]
    idx_matched_obj2 = idx_obj2[idx_matched_pairs]
    # Getting also deltaR and maskDR of the valid pairs
    deltaR_matched = deltaR_sorted[idx_matched_pairs]
    maskDR_matched = maskDR_sorted[idx_matched_pairs]



    # We get the indices needed to reorder the second collection
    # and we use them to re-order also the other collection (same dimension of the valid pairs)
    obj2_order = ak.argsort(idx_matched_obj2)
    idx_obj_obj2sorted = idx_matched_obj[obj2_order]
    idx_obj2_obj2sorted = idx_matched_obj2[obj2_order]
    deltaR_obj2sorted = deltaR_matched[obj2_order]
    maskDR_obj2sorted = maskDR_matched[obj2_order]
    # Here we apply the deltaR + pT requirements on the objects and on deltaR
    idx_obj_masked = idx_obj_obj2sorted[maskDR_obj2sorted]
    idx_obj2_masked = idx_obj2_obj2sorted[maskDR_obj2sorted]
    # Getting also the deltaR of the masked pairs
    deltaR_masked = deltaR_obj2sorted[maskDR_obj2sorted]
    # N.B. We are still working only with indices not final objects

    # Now we have the object in the collection 1 ordered as the collection 2,
    # but we would like to have an ak.Array of the dimension of the collection 2, with "None"
    # in the places where there is not matching.
    # We need a special function for that, building the ak.Array of object from collection 1, with the dimension of collection 2, with None padding.
    (
        _idx_obj_padnone,
        _idx_obj2_padnone,
        _deltaR_padnone,
    ) = get_matching_objects_indices_padnone(
        ak.without_parameters(idx_obj_masked, behavior={}),
        ak.without_parameters(idx_obj2_masked, behavior={}),
        ak.without_parameters(ak.num(obj2), behavior={}),
        ak.without_parameters(deltaR_masked, behavior={}),
        ak.ArrayBuilder(),
        ak.ArrayBuilder(),
        ak.ArrayBuilder(),
    )
    idx_obj_padnone = _idx_obj_padnone.snapshot()
    idx_obj2_padnone = _idx_obj2_padnone.snapshot()
    deltaR_padnone = _deltaR_padnone.snapshot()

    # Finally the objects are sliced through the padded indices
    # In this way, to a None entry in the indices will correspond a None entry in the object
    matched_obj = obj1[idx_obj_padnone]
    matched_obj2 = obj2[idx_obj2_padnone]

    return (
        matched_obj,
        matched_obj2,
        deltaR_padnone,
        idx_obj_padnone,
        idx_obj2_padnone,
        deltaR_masked
    )


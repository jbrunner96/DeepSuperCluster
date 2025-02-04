import awk_data
import tensorflow as tf
import argparse 
from collections import defaultdict
import os, json
import importlib.util
from time import time
import numpy as np
import plot_loss
import gc

parser = argparse.ArgumentParser()

parser.add_argument("--config", type=str, help="Config", required=True)
parser.add_argument("--model", type=str, help="Model .py", required=True)
parser.add_argument("--name", type=str, help="Model version name", required=False)
parser.add_argument("--output", type=str,help="Override output folder", required=False)
parser.add_argument("--debug", action="store_true", help="Debug and run TF eagerly")
parser.add_argument("--profile", action="store_true", help="Profile model training")
parser.add_argument("-v","--verbose", action="store_true", help="Verbose")
args = parser.parse_args()

config = json.load(open(args.config))


# Checking hardware
print('version={}, CUDA={}, GPU={}'.format(
    tf.__version__, tf.test.is_built_with_cuda(),
    len(tf.config.list_physical_devices('GPU')) > 0))
      
gpus =  tf.config.list_physical_devices('GPU')
print("gpus: ", gpus)

num_threads = 3 # Why 5?
os.environ["OMP_NUM_THREADS"] = str(num_threads)
os.environ["TF_NUM_INTRAOP_THREADS"] = str(num_threads)
os.environ["TF_NUM_INTEROP_THREADS"] = str(num_threads)

tf.config.threading.set_inter_op_parallelism_threads(
    num_threads
)
tf.config.threading.set_intra_op_parallelism_threads(
    num_threads
)
    
if len(gpus) >=1 :
    print("Using 1 GPU")
    # tf.config.experimental.set_memory_growth(gpus[0], enable=True)
    print("Using 1 GPU: ", os.environ['CUDA_VISIBLE_DEVICES'])
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    strategy = tf.distribute.OneDeviceStrategy(f"GPU:0")
# elif len(gpus):
#     print("Using {} GPUs".format(len(gpus)))
#     for gpu in gpus:
#         tf.config.experimental.set_memory_growth(gpu, enable=True)
#     strategy = tf.distribute.MirroredStrategy()-
else:
    tf.config.set_soft_device_placement(True)
    strategy = tf.distribute.OneDeviceStrategy("cpu:0")

##################
# Prepare the output folder
def get_unique_run():
    previous_runs = list(filter( lambda k: k.startswith("run"), os.listdir(config["models_path"])))
    run_number = 1
    if len(previous_runs) > 0:
        run_number = max([int(s.split('_')[1]) for s in previous_runs if s.startswith("run")]) + 1 
    return run_number


if args.output != None:
    config["models_path"] = args.output
if not os.path.isdir(config["models_path"]):
    os.makedirs(config["models_path"])

name =  f'run_{get_unique_run():02}'
if args.name != None:
    name += f"_{args.name}"

outdir = config["models_path"] + "/"+ name
config["model_name"] = name 

if os.path.isdir(outdir):
    print("Output directory exists: {}".format(outdir), file=sys.stderr)
else:
    os.makedirs(outdir)

print("Model output folder: ", outdir)
# Save the output folder in the config
config["models_path"] = outdir
config["model_definition_path"] = os.path.join(outdir, os.path.split(args.model)[-1])
#Copying the config file and model file in the output dir:
os.system("cp {} {}".format(args.model, outdir))
json.dump(config, open(os.path.join(outdir, "training_config.json"),"w"),
          indent=2)


###########################
## Loading the datasets
print(">>> Loading datasets")

ds_train = awk_data.load_dataset(awk_data.LoaderConfig(**config["dataset_conf"]["training"]))
ds_test = awk_data.load_dataset(awk_data.LoaderConfig(**config["dataset_conf"]["validation"]))
# Create training and validation
# ds_train = train_ds.prefetch(tf.data.AUTOTUNE).repeat(config['nepochs'])
# ds_test  = test_ds.prefetch(tf.data.AUTOTUNE).repeat(config['nepochs'])
#ds_train = train_ds.repeat(config['nepochs'])
#ds_test  = test_ds.repeat(config['nepochs'])

ds_train = ds_train.map(lambda x,y,z: (x,y,z), num_parallel_calls=1)
ds_test = ds_test.map(lambda x,y,z: (x,y,z), num_parallel_calls=1)

############### 
# Loading the model file
 # Load model modules
spec = importlib.util.spec_from_file_location("model", args.model)
model_lib = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model_lib)

config['activation'] = tf.keras.activations.get(config['activation'])

tf.keras.backend.clear_session()
# Construction of the model in the strategy scope
with strategy.scope():
    print(">>> Creating the model")
    # Build the model with all the configs
    model = model_lib.DeepClusterGN(**config)

    #optimizer
    if config['opt'] == "adam":
        opt = tf.keras.optimizers.Adam(learning_rate=config['lr'])
    elif config['opt'] == "adamW":
        opt = tf.keras.optimizers.AdamW(learning_rate=config['lr'])
        
    #compile the model
    model.set_metrics()

    model.compile(optimizer=opt,
                  run_eagerly=args.debug,
                  weighted_metrics=[])

    for X, y ,w  in ds_train:
        # Load the model
        ypred = model(X, training=False)
        #l = custom_loss(y, ypred)
        break

    model.summary()
    
    # Callback
    callbacks = []

    terminate_cb = tf.keras.callbacks.TerminateOnNaN()
    callbacks += [terminate_cb]

    if "lr_reduce" in config:
        lr_reduce = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=config["lr_reduce"]["factor"], patience=config["lr_reduce"]["patience"], verbose=1,
            mode='auto', min_delta=config["lr_reduce"]["min_delta"], cooldown=0, min_lr=config["lr_reduce"].get("min",1e-7),
        )
        callbacks.append(lr_reduce)

    # cp_callback = tf.keras.callbacks.ModelCheckpoint(
    #     filepath=outdir + "/weights.{epoch:02d}-{val_loss:.6f}.hdf5",
    #     save_weights_only=True,
    #     verbose=1
    # )
    # cp_callback.set_model(model)
    # callbacks.append(cp_callback)


    # if "early_stop" in config:
    #     early = tf.keras.callbacks.EarlyStopping(
    #         monitor='val_loss', min_delta=config["early_stop"]["min_delta"], patience=config["early_stop"]["patience"],
    #          verbose=1, mode='auto', baseline=None, restore_best_weights=False
    #     )
    #     callbacks.append(early)

    # if config["loss_plot"]:
    #     loss_plotter = plot_loss.LossPlotter(outdir, batch_mode=True)
    #     callbacks.append(loss_plotter)

    # if args.profile:
    #     tb_callback = tf.keras.callbacks.TensorBoard(outdir+"/profiler_logs",
    #                                              profile_batch=(50, 10100),
    #                                              update_freq='batch')
    #     callbacks.append(tb_callback)

    class ClearMemoryCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            gc.collect()
            tf.keras.backend.clear_session()

    callbacks.append(ClearMemoryCallback())
            
    if "comet" in config:
        # do not import if not needed
        import comet_ml
        experiment = comet_ml.Experiment(
                api_key=config["comet"]["api_key"],
                project_name=config["comet"]["project_name"],
                workspace=config["comet"]["workspace_name"]
        )
        experiment.set_name(name)
        comet_callback = experiment.get_callback("keras")
        callbacks.append(comet_callback)



    if args.verbose:
        verbosity = 1
    else:
        verbosity = 2
    # FINALLY TRAINING!
    # print(">>> Start training")
    # history = model.fit(ds_train,
    #     validation_data=ds_test, 
    #     epochs=config['nepochs'],
    #     steps_per_epoch= config['dataset_conf']["training"]["maxevents"]//config['dataset_conf']["training"]['batch_size'], 
    #     validation_steps= config['dataset_conf']["validation"]["maxevents"]//config['dataset_conf']["validation"]['batch_size'],
    #     verbose=verbosity,
    #     callbacks = callbacks
    # )

    # with open(outdir+"/training_history.csv",'w') as of:
    #     for key in history.history.keys():
    #         of.write(key+ ";")
    #     of.write('\n')
    #     n = len(history.history['loss'])
    #     for i in range(n):
    #         for key in history.history.keys():
    #             of.write(str(history.history[key][i])+';')
    #         of.write('\n')
    
    # print(">>> Done!")

training_data_iter = iter(ds_train)
validation_data_iter = iter(ds_test)

    
# Training loop
for epoch in range(config['nepochs']):
    print(f'Starting epoch {epoch}')

    # reset the mtrics
    metrics = model.metrics
    for m in metrics:
        m.reset_states()

    # Iterate over the batches of the dataset.
    for step in range(config['dataset_conf']["training"]["maxevents"]//config['dataset_conf']["training"]['batch_size']):
        x_batch_train = next(training_data_iter)
        out = model.train_step(x_batch_train)

        # Print out the metrics for this epoch
        print(f'step {step}: {out}')

    for step in range(config['dataset_conf']["validation"]["maxevents"]//config['dataset_conf']["validation"]['batch_size']):
        x_batch_val = next(validation_data_iter)
        model.test_step(x_batch_val)

        

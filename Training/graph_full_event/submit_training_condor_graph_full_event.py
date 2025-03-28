import htcondor
import argparse
import os

col = htcondor.Collector()
credd = htcondor.Credd()
credd.add_user_cred(htcondor.CredTypes.Kerberos, None)


parser = argparse.ArgumentParser()
parser.add_argument("--basedir", type=str, help="Base dir", default=os.getcwd())
parser.add_argument("--name", type=str, help="Model version name", required=False, default="base")
parser.add_argument("--config", type=str, help="config file (relative to base dir)", required=True)
parser.add_argument("--model", type=str, help="Model.py (relative to basedir)", required=True)
parser.add_argument("--test", action="store_true", help="Do no run condor job but interactively")
parser.add_argument("--venv", type=str, help="Virtual environment", required=True, default="None")
parser.add_argument("--apikey", type=str, help="comet API key", required=False)
args = parser.parse_args()

# Checking the input files exists
os.makedirs(args.basedir, exist_ok=True)
os.makedirs(f"{args.basedir}/condor_logs", exist_ok=True)
os.makedirs(f"{args.basedir}/condor_logs/output", exist_ok=True)
os.makedirs(f"{args.basedir}/condor_logs/error", exist_ok=True)
os.makedirs(f"{args.basedir}/condor_logs/log", exist_ok=True)

if not os.path.exists(os.path.join(args.basedir, args.config)):
    raise ValueError(f"Config file does not exists: {args.config}")
if not os.path.exists(os.path.join(args.basedir, args.model)):
    raise ValueError(f"Model file does not exists: {args.model}")

# run without condor
if args.test:
    os.system(f"sh run_training_condor_graph_full_event.sh {args.basedir}/{args.config} {args.basedir}/{args.model} {args.name} {args.venv} {args.apikey}")
    exit(0)


sub = htcondor.Submit()
sub['Executable'] = "run_training_condor_graph_full_event.sh"
sub["arguments"] = f"{args.basedir}/{args.config}  {args.basedir}/{args.model} {args.name} {args.venv} {args.apikey}"
sub['Error'] = args.basedir+"/condor_logs/error/training-$(ClusterId).$(ProcId).err"
sub['Output'] = args.basedir+"/condor_logs/output/training-$(ClusterId).$(ProcId).out"
sub['Log'] = args.basedir+"/condor_logs/log/training-$(ClusterId).log"
sub['MY.SendCredential'] = True
sub['+JobFlavour'] = '"testmatch"'
sub["transfer_input_files"] = "trainer_graph_full_event.py, dataset_loader.py, dataset_utils.py"
sub["when_to_transfer_output"] = "ON_EXIT"
sub['request_cpus'] = '6'
sub['request_gpus'] = '1'

schedd = htcondor.Schedd()
with schedd.transaction() as txn:
    cluster_id = sub.queue(txn)
    print(cluster_id)
    # Saving the log
    with open(f"{args.basedir}/condor_logs/training_jobs.csv", "a") as l:
        l.write(f"{cluster_id};{args.name};{args.model};{args.config};{args.basedir}\n")

    

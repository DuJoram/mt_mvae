import os
import json
import tempfile

prep_leomed = """
echo \"Loading Modules...\"
module load cuda/10.0.130
module load cudnn/7.5
module load openblas/0.2.19
"""
str_conda = """
eval \"$(conda shell.bash hook)\"
conda activate joint_elbo
"""

str_tmpdir = """
if [[ -z "$TMPDIR" ]]; then
TMPDIR="/tmp"
fi
echo "TMPDIR: $TMPDIR"
"""

def read_base_json(fn):
    f_json = open(fn, "r")
    d_json = json.load(f_json)
    f_json.close()
    return d_json


def create_tmp_dir():
    if not tempfile.gettempdir():
        tmpdirname = '/tmp'
        with tempfile.TemporaryDirectory() as tmpdirname:
            print('created temporary directory', tmpdirname)


def get_cp_data_str(params, dataset):
    fn_data = os.path.join(params['dir_data'], 'PolyMNIST_' + dataset + '.zip')
    str_data = ''
    if os.path.exists(params['unimodal-datapaths-train']):
        pass
    else:
        tmpDir = tempfile.gettempdir()
        str_cp = 'cp -r ' + fn_data + ' ' + '\"${TMPDIR}/\"'
        str_data = str_data + str_cp + '\n'
        str_data = str_data + 'cd ' + '\"${TMPDIR}/\"' + '\n'
        str_data = str_data + 'unzip -q PolyMNIST_' + dataset + '.zip' + '\n'
        str_data = str_data + 'cd -' + '\n'
    return str_data


def write_job_script(params, fn, device, dataset="vanilla"):
    create_tmp_dir()
    f_out = open(fn, 'w')
    f_out.write("#!/usr/bin/env bash" + "\n")
    f_out.write("set -eo pipefail \n")
    f_out.write("shopt -s nullglob globstar \n")
    f_out.write("\n")

    if device == 'leomed':
        f_out.write(prep_leomed)
        str_data = get_cp_data_str(params, dataset)
        f_out.write(str_data)
        f_out.write('\n')

    f_out.write(str_conda + "\n")
    f_out.write(str_tmpdir + "\n")

    f_out.write("python main_mmnist_" + dataset + ".py ")
    for _, (key, value) in enumerate(params.items()):
        if value is False or value is True:
            if value is True:
                f_out.write("--" + str(key) + " \\" + "\n")
            else:
                continue
        elif (key.startswith('unimodal-datapaths') or
              key.startswith('pretrained-classifier-paths')):
            f_out.write("--" + str(key) + " " + str(value) + " \\" + "\n")
        else:
            f_out.write("--" + str(key) + "=" + str(value) + " \\" + "\n")
    f_out.close()


if __name__ == '__main__':
    fn_json = 'run_files/params_base.json'
    d_args = read_base_json(fn_json)
    print(d_args)
    fn_out = 'job_test.1'
    write_job_script(d_args, fn_out, 'leomed')

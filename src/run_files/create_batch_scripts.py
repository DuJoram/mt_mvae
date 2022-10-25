import os
import subprocess

from create_run_file import read_base_json
from create_run_file import write_job_script


str_bsub = 'bsub -n 4 -W 24:00 -R \"rusage[mem=16000,scratch=10000,ngpus_excl_p=1]\" bash'


def create_batch_runs(l_names, l_values, fn_base, fn_out_base, dataset, str_exp_run=None):
    values_p1 = l_values[0]
    name_p1 = l_names[0]

    for v_p1 in values_p1:
        d_args = read_base_json(fn_base)
        d_args[name_p1] = v_p1
        name_method = str(d_args['method'][1:-1])
        str_run = name_method + '_' + name_p1 + str(v_p1)
        if str_exp_run is not None:
            dir_exp = os.path.join(d_args['dir_experiment'], str_exp_run)
            d_args['dir_experiment'] = dir_exp
        fn_out = fn_out_base + str_run
        write_job_script(d_args, fn_out, 'leomed', dataset)
        bashCmd = str_bsub + ' ' + fn_out
        print(bashCmd)
        process = subprocess.Popen(bashCmd,
                                   shell=True,
                                   stdout=subprocess.PIPE)
        output, error = process.communicate()


if __name__ == '__main__':
    dataset = 'vanilla'
    name_p1 = 'beta'
    values_p1 = [0.0001, 0.001, 0.01, 0.1, 0.5, 1.0, 2.5, 5.0, 10.0]
    list_names = [name_p1]
    list_values = [values_p1]
    fn_out_base = os.path.join('run_scripts', 'job_batch.')
    fn_base_args = os.path.join('run_files', 'params_base_leomed_PolyMNIST' + dataset + '.json')
    str_run = 'beta_0.0001_10.0_unw'
    create_batch_runs(list_names, list_values, fn_base_args, fn_out_base,
                      dataset, str_exp_run=str_run)


import os
import numpy as np
import torch
from glob import glob
from utils import misc
import argparse
from utils.evaluation.similarity import tanimoto_sim


def get_rfn(r):
    fn_name = r['ligand_filename'].split('/')[0] + r['ligand_filename'].split('/')[1].split('_')[0]  + r['ligand_filename'].split('/')[1].split('_')[1]  + r['ligand_filename'].split('/')[1].split('_')[2]  + r['ligand_filename'].split('/')[1].split('_')[3]  + r['ligand_filename'].split('/')[1].split('_')[4]
    return fn_name

def calc_pairwise_sim(mols):
    n = len(mols)
    sims = []
    for i in range(n):
        for j in range(i + 1, n):
            sims.append(tanimoto_sim(mols[i], mols[j]))
    return np.array(sims)


def computer_diversity(results):
    div_all = []
    # for result in tqdm(results):
    for result in results:
        mols = [r['mol'] for r in result]
        div_all.append(np.mean(1 - calc_pairwise_sim(mols)))

    div_all = np.array(div_all)
    div_all = div_all[~np.isnan(div_all)]
    print('Diversity: Mean: %.4f Median: %.4f' % (np.mean(div_all), np.median(div_all)))


def spilt_results_by_pocket_fn(results_all, results_fn_all, ref_fn_all):
    # results_all = copy.copy(results_all_old)
    num_all_results = len(results_all)
    num_all_results_sub = 0
    num_ref_fn_all = len(ref_fn_all)
    split_results_all = []
    for rfn in ref_fn_all:
        sub_results = []
        for r in results_all:
            if r['ligand_filename'].split('/')[0] + r['ligand_filename'].split('/')[1].split('_')[0]  + r['ligand_filename'].split('/')[1].split('_')[1]  + r['ligand_filename'].split('/')[1].split('_')[2]  + r['ligand_filename'].split('/')[1].split('_')[3]  + r['ligand_filename'].split('/')[1].split('_')[4] == rfn:
                sub_results.append(r)

        split_results_all.append(sub_results)
        num_all_results_sub += len(sub_results)
    return split_results_all

def compute_ref_success_rate(vina_ref_dict, qed_ref_dict, sa_ref_dict, split_results, mode='dock'):
    percentage_good = []
    total_num = 0
    success_num = 0

    for i in range(100):
        
        pocket_results = split_results[i]


        rfn = get_rfn(pocket_results[0])

        score_ref = vina_ref_dict[rfn]
        qed_ref = vina_ref_dict[rfn]
        sa_ref = vina_ref_dict[rfn]
        if len(pocket_results) < 50:
            continue
        # num_docked.append(len(pocket_results))
        total_num += len(pocket_results)

        scores_gen = []
        for docked in pocket_results:
            aff = docked['vina'][mode][0]['affinity']
            qed = docked['chem_results']['qed']
            sa = docked['chem_results']['sa']

            if qed >= qed_ref and sa >=sa_ref and aff <= score_ref :
                success_num += 1

    success_rate = success_num * 1.0 / total_num
    print('Ref Success Rate: %.4f at %d' % (success_rate * 100, total_num))



def compute_high_affinity(vina_ref_dict, split_results, mode='dock'):
    percentage_good = []
    # num_docked = []
    qed_good, sa_good = [], []
    for i in range(100):
        
        pocket_results = split_results[i]
        rfn = get_rfn(pocket_results[0])
        score_ref = vina_ref_dict[rfn]
        if len(pocket_results) < 50:
            continue
        # num_docked.append(len(pocket_results))

        scores_gen = []
        for docked in pocket_results:
            aff = docked['vina'][mode][0]['affinity']
            scores_gen.append(aff)
            if aff <= score_ref:
                qed_good.append(docked['chem_results']['qed'])
                sa_good.append(docked['chem_results']['sa'])
        scores_gen = np.array(scores_gen)
        percentage_good.append((scores_gen <= score_ref).mean())

    percentage_good = np.array(percentage_good)
    # num_docked = np.array(num_docked)
    # print('valid pockets: ', len(num_docked), sum(num_docked))
    print('High Affinity%%: Mean: %.2f%% Median: %.2f%% ' % (np.mean(percentage_good) * 100, np.median(percentage_good) * 100))
    # print('[HF-QED]  Avg: %.4f | Med: %.4f ' % (np.mean(qed_good) * 100, np.median(qed_good) * 100))
    # print('[HF-SA]   Avg: %.4f | Med: %.4f ' % (np.mean(sa_good) * 100, np.median(sa_good) * 100))
    # print('[Success%%] %.2f%% ' % (np.mean(percentage_good > 0)*100, ))


def compute_success_rate(vina_ref_dict, split_results, mode='dock'):
    # print("len(split_results) -> ", len(split_results))
    # num_docked = []
    total_num = 0
    success_num = 0
    for i in range(100):
        
        pocket_results = split_results[i]
        rfn = get_rfn(pocket_results[0])
        score_ref = vina_ref_dict[rfn]
        if len(pocket_results) < 50:
            continue
        # num_docked.append(len(pocket_results))
        total_num += len(pocket_results)

        scores_gen = []

        for docked in pocket_results:
            aff = docked['vina'][mode][0]['affinity']
            qed = docked['chem_results']['qed']
            sa = docked['chem_results']['sa']
            QED = 0.25
            SA = 0.59
            Vina_Dock = -8.18

            if qed >= QED and sa >=SA and aff <= Vina_Dock :
                success_num += 1

    success_rate = success_num * 1.0 / total_num
    print('Success Rate: %.4f at %d' % (success_rate * 100, total_num))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_result_path', default='/home/huangzl/Data/workspace/self-gen/SG-Self-IRDiff_db128_n1r1_sconly_nb_gsbapv3-n05-bit_rb-ipdiff_chunkcfg-scale075/eval_results/', type=str)
    parser.add_argument('--docking_mode', type=str, default='vina_dock', choices=['none', 'vina_score', 'vina_dock'])
    args = parser.parse_args()

    eval_path = args.eval_result_path
    docking_mode = args.docking_mode
    results_fn_list = glob(os.path.join(eval_path, 'metrics_*.pt'))

    logger = misc.get_logger('evaluation_overall', log_dir=eval_path)

    logger.info(f'the num of results.pt is {len(results_fn_list)}.')

    qed_all = []
    sa_all = []
    vina_score_all = []
    vina_min_all = []
    vina_dock_all = []
    ligand_fn_all = []
    pocket_fn_all = []
    results_all = []

    for rfn in results_fn_list:
        result_i = torch.load(rfn)['all_results']
        qed_all += [r['chem_results']['qed'] for r in result_i]
        sa_all += [r['chem_results']['sa'] for r in result_i]
        ligand_fn_all += [r['ligand_filename'] for r in result_i]
        pocket_fn_all += [r['ligand_filename'].split('/')[0] for r in result_i]
        results_all += [r for r in result_i]

        if docking_mode in ['vina_dock', 'vina_score']:
            vina_score_all += [r['vina']['score_only'][0]['affinity'] for r in result_i]
            vina_min_all += [r['vina']['minimize'][0]['affinity'] for r in result_i]
            if docking_mode == 'vina_dock':
                vina_dock_all += [r['vina']['dock'][0]['affinity'] for r in result_i]

    print("all nums of results : ", len(results_all))
    qed_all_mean, qed_all_median = np.mean(qed_all), np.median(qed_all)
    sa_all_mean, sa_all_median = np.mean(sa_all), np.median(sa_all)

    print('QED:   Mean: %.3f Median: %.3f' % (qed_all_mean, qed_all_median))
    print('SA:    Mean: %.3f Median: %.3f' % (sa_all_mean, sa_all_median))

    if len(vina_score_all):
        vina_score_all_mean, vina_score_all_median = np.mean(vina_score_all), np.median(vina_score_all)
        print('Vina Score:  Mean: %.3f Median: %.3f' % (vina_score_all_mean, vina_score_all_median))

    if len(vina_min_all):
        vina_min_all_mean, vina_min_all_median = np.mean(vina_min_all), np.median(vina_min_all)
        print('Vina Min:  Mean: %.3f Median: %.3f' % (vina_min_all_mean, vina_min_all_median))

    if len(vina_dock_all):
        vina_dock_all_mean, vina_dock_all_median = np.mean(vina_dock_all), np.median(vina_dock_all)
        print('Vina Dock:  Mean: %.3f Median: %.3f' % (vina_dock_all_mean, vina_dock_all_median))

    ### Calculate high-affinity

    reference_results = torch.load('/home/huangzl/Data/datasets/molecule/crossdocked_test_vina_docked.pt')
    # vina_ref = [r['vina']['dock'][0]['affinity'] for r in reference_results]
    vina_ref_dict = {}
    qed_ref_dict = {}
    sa_ref_dict = {}
    for r in reference_results:
        dockvalue = r['vina']['dock'][0]['affinity']
        qedvalue = r['chem_results']['qed']
        savalue = r['chem_results']['sa']
        rn = get_rfn(r)
        vina_ref_dict[rn] = dockvalue
        qed_ref_dict[rn] = qedvalue
        sa_ref_dict[rn] = savalue
    assert len(vina_ref_dict.keys()) == len(reference_results)
    assert len(qed_ref_dict.keys()) == len(reference_results)
    assert len(sa_ref_dict.keys()) == len(reference_results)

    ref_pocket_fn_all = [get_rfn(r) for r in reference_results]

    assert len(ref_pocket_fn_all) == 100

    split_results_all = spilt_results_by_pocket_fn(results_all=results_all, results_fn_all=pocket_fn_all,
                                                   ref_fn_all=ref_pocket_fn_all)

    compute_high_affinity(vina_ref_dict, split_results_all, mode='dock')
    compute_success_rate(vina_ref_dict, split_results_all, mode='dock')
    compute_ref_success_rate(vina_ref_dict, qed_ref_dict, sa_ref_dict, split_results_all, mode='dock')
    computer_diversity(split_results_all)





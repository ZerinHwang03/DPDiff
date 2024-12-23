import torch
from glob import glob
import os

if __name__ == '__main__':
    dir_path = '/userhome/workspace/BindFusion-fullatom/bindfusion-ra-augsc/bindfusionra-augsc_padd-lla-fl_intgat-f10coN/sampled_results_21'
    results_fp_list = glob(os.path.join(dir_path, '*.pt'))

    for f_path in results_fp_list:
        f = torch.load(f_path)


        # dict_keys(['data', 'pred_ligand_pos', 'pred_ligand_v', 'pred_ligand_pos_traj', 'pred_ligand_v_traj', 'time'])

        # print('data -> ', f['data'])
        # print('pred_ligand_pos -> ', f['pred_ligand_pos'][0].shape)
        # print('pred_ligand_v -> ', f['pred_ligand_v'][0].shape)
        # print('pred_ligand_pos_traj -> ', f['pred_ligand_pos_traj'][0].shape)
        # print('pred_ligand_v_traj -> ', f['pred_ligand_v_traj'][0].shape)

        new_f = {}
        new_f['data'] = f['data']
        new_f['pred_ligand_pos'] = f['pred_ligand_pos']
        new_f['pred_ligand_v'] = f['pred_ligand_v']
        # for n samples
        # print(type(f['pred_ligand_pos_traj']))
        # print(type(f['pred_ligand_pos_traj'][0]))
        assert type(f['pred_ligand_pos_traj']) == type([1, 2,])

        new_f['pred_ligand_pos_traj'] = [fi[-1] for fi in f['pred_ligand_pos_traj']]
        new_f['pred_ligand_v_traj'] = [fi[-1] for fi in f['pred_ligand_v_traj']]

        # print('data -> ', new_f['data'])
        # print('new_fpred_ligand_pos -> ', new_f['pred_ligand_pos'][0].shape)
        # print('new_fpred_ligand_v -> ', new_f['pred_ligand_v'][0].shape)
        # print('new_fpred_ligand_pos_traj -> ', new_f['pred_ligand_pos_traj'][0].shape)
        # print('new_fpred_ligand_v_traj -> ', new_f['pred_ligand_v_traj'][0].shape)

        # torch.save(new_f, './new_%d' % data_id)
        new_f_path, shuffix = f_path.split('.')
        new_f_path = new_f_path + '_lite_.' + shuffix
        torch.save(new_f, new_f_path)
        print('save in new path: ', new_f_path)



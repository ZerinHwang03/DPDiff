import argparse
import os
import shutil

import numpy as np
import torch
# import torch.utils.tensorboard
from sklearn.metrics import roc_auc_score
from torch.nn.utils import clip_grad_norm_
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose
from tqdm.auto import tqdm

import utils.misc as misc
import utils.train as utils_train
import utils.transforms as trans
from datasets import get_topk_promt_dataset
from datasets.pl_data import FOLLOW_BATCH
from models.molopt_score_model import ScorePosNet3D

from graphbap.normal_bapnet import Normal_BAPNet
from graphbap.relative_bapnet import Relative_BAPNet

def get_auroc(y_true, y_pred, feat_mode):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    avg_auroc = 0.
    possible_classes = set(y_true)
    for c in possible_classes:
        auroc = roc_auc_score(y_true == c, y_pred[:, c])
        avg_auroc += auroc * np.sum(y_true == c)
        mapping = {
            'basic': trans.MAP_INDEX_TO_ATOM_TYPE_ONLY,
            'add_aromatic': trans.MAP_INDEX_TO_ATOM_TYPE_AROMATIC,
            'full': trans.MAP_INDEX_TO_ATOM_TYPE_FULL
        }
        print(f'atom: {mapping[feat_mode][c]} \t auc roc: {auroc:.4f}')
    return avg_auroc / len(y_true)


if __name__ == '__main__':
    root_dir = '/userhome/huangzhl/workspace_sg/SG-Self-IRDiff_db128_n1r1_auglla_nb-gsbapv3-n05-bit_rb-ipdiff_chunkcfg-scale05'

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=root_dir + '/configs/training.yml')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--logdir', type=str, default=root_dir + '/logs')
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--train_report_iter', type=int, default=200)
    args = parser.parse_args()

    # Load configs
    config = misc.load_config(args.config)
    config_name = os.path.basename(args.config)[:os.path.basename(args.config).rfind('.')]
    misc.seed_all(config.train.seed)

    # Logging
    log_dir = misc.get_new_log_dir(args.logdir, prefix=config_name, tag=args.tag)
    ckpt_dir = os.path.join(log_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    vis_dir = os.path.join(log_dir, 'vis')
    os.makedirs(vis_dir, exist_ok=True)
    logger = misc.get_logger('train', log_dir)
    # writer = torch.utils.tensorboard.SummaryWriter(log_dir)
    logger.info(args)
    logger.info(config)
    shutil.copyfile(args.config, os.path.join(log_dir, os.path.basename(args.config)))
    shutil.copytree(root_dir + '/models', os.path.join(log_dir, 'models'))

    # -------- start: data preparation --------
    # Transforms
    protein_featurizer = trans.FeaturizeProteinAtom()
    ligand_featurizer = trans.FeaturizeLigandAtom(config.data.transform.ligand_atom_mode)
    transform_list = [
        protein_featurizer,
        ligand_featurizer,
        trans.FeaturizeLigandBond(),
    ]
    if config.data.transform.random_rot:
        transform_list.append(trans.RandomRotation())
    transform = Compose(transform_list)

    # -------- end: data preparation --------

    # Datasets and loaders
    # -------- start: config dataloader --------
    logger.info('Loading dataset...')
    # dataset, subsets = get_dataset(
    #     config=config.data,
    #     transform=transform,
    # )
    # subsets = get_promt_dataset(
    #     config=config.data,
    #     transform=transform,
    # )

    subsets = get_topk_promt_dataset(
        config=config.data,
        transform=transform,
    )
    # Todo: relative_train_prompts
    relative_train_prompts_path = '/userhome/huangzhl/datasets/sbdd/self-gen-ipdiff/final_mols_dataset.pt'
    relative_train_prompts = torch.load(relative_train_prompts_path, map_location='cpu')
    relative_train_prompts_convert_index_path = '/userhome/huangzhl/datasets/sbdd/self-gen-ipdiff/train_relative_prompts_convert_indices.pt'
    relative_train_prompts_convert_index = torch.load(relative_train_prompts_convert_index_path, map_location='cpu')
    relative_test_prompts_convert_index_path = '/userhome/huangzhl/datasets/sbdd/self-gen-ipdiff/test_relative_prompts_convert_indices.pt'
    relative_test_prompts_convert_index = torch.load(relative_test_prompts_convert_index_path, map_location='cpu')
    normal_topk_prompt = config.data.normal_topk_prompt

    train_set, val_set = subsets['train'], subsets['test']
    logger.info(f'Training: {len(train_set)} Validation: {len(val_set)}')

    # follow_batch = ['protein_element', 'ligand_element']
    collate_exclude_keys = ['ligand_nbh_list']
    train_iterator = utils_train.inf_iterator(DataLoader(
        train_set,
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=config.train.num_workers,
        follow_batch=FOLLOW_BATCH,
        exclude_keys=collate_exclude_keys
    ))
    val_loader = DataLoader(val_set, config.train.val_batch_size, shuffle=False,
                            follow_batch=FOLLOW_BATCH, exclude_keys=collate_exclude_keys)
    # -------- end: config dataloader --------

    # -------- start: build model --------
    # Model
    logger.info('Building model...')

    ## [BF] introduce pretrained BAPNet
    normal_net_cond = Normal_BAPNet(ckpt_path=config.normal_net_cond.ckpt_path, hidden_nf=config.normal_net_cond.hidden_dim).to(args.device)
    relative_net_cond = Relative_BAPNet(ckpt_path=config.relative_net_cond.ckpt_path, hidden_nf=config.relative_net_cond.hidden_dim).to(args.device)

    model = ScorePosNet3D(
        config.model,
        protein_atom_feature_dim=protein_featurizer.feature_dim,
        ligand_atom_feature_dim=ligand_featurizer.feature_dim
    ).to(args.device)
    # -------- end: build model --------

    print(f'protein feature dim: {protein_featurizer.feature_dim} ligand feature dim: {ligand_featurizer.feature_dim}')
    logger.info(f'# trainable parameters: {misc.count_parameters(model) / 1e6:.4f} M')

    # Optimizer and scheduler
    optimizer = utils_train.get_optimizer(config.train.optimizer, model)
    scheduler = utils_train.get_scheduler(config.train.scheduler, optimizer)

    # -------- start: start training --------
    def train(it):
        model.train()
        optimizer.zero_grad()
        for _ in range(config.train.n_acc_batch):
            all_batch = next(train_iterator)
            all_batch = [b.to(args.device) for b in all_batch]
            assert len(all_batch) == normal_topk_prompt + 2, "wrong value of topk_prompt"

            if normal_topk_prompt == 1:
                batch, normal_prompt_batch, train_data_idx = all_batch
            elif normal_topk_prompt == 2:
                batch, normal_prompt_batch, normal_prompt_batch_2, train_data_idx = all_batch
            elif normal_topk_prompt == 3:
                batch, normal_prompt_batch, normal_prompt_batch_2, normal_prompt_batch_3, train_data_idx = all_batch
            else:
                raise ValueError(normal_topk_prompt)

            # Todo: relative_topk_prompt
            device = args.device
            train_data_idx = train_data_idx.unsqueeze(0).to('cpu')
            while len(train_data_idx.shape) > 1:
                train_data_idx = train_data_idx.squeeze(0)
            while len(train_data_idx.shape) == 0:
                train_data_idx = train_data_idx.unsqueeze(0)
            train_data_idx_list = [train_data_idx[i].item() for i in range(train_data_idx.shape[0])]
            train_data_idx_list_convert = [relative_train_prompts_convert_index[i] for i in train_data_idx_list]
            bs = len(train_data_idx_list)

            # ['data', 'pred_ligand_pos', 'pred_ligand_v']
            relative_prompt_batch_pos = torch.cat([torch.tensor(
                relative_train_prompts[train_data_idx_list_convert[bi]]['pred_ligand_pos'][0], dtype=torch.float32) for bi in range(bs)], dim=0).to(device)
            relative_prompt_batch_v = torch.cat([torch.tensor(relative_train_prompts[train_data_idx_list_convert[bi]]['pred_ligand_v'][0]) for bi in range(bs)], dim=0).long().squeeze().to(device)
            # relative_prompt_batch_batch = torch.repeat_interleave(torch.arange(bs), torch.tensor(relative_train_prompts['num_atom'])).to(device)
            relative_prompt_batch_batch_1_list = []
            for bi in range(bs):
                relative_prompt_batch_batch_1_list += [torch.tensor(bi)] * relative_train_prompts[train_data_idx_list_convert[bi]]['pred_ligand_v'][0].shape[0]
            relative_prompt_batch_batch = torch.tensor(relative_prompt_batch_batch_1_list).long().to(device)

            relative_prompt_batch_pos_2 = torch.cat([torch.tensor(relative_train_prompts[train_data_idx_list_convert[bi]]['pred_ligand_pos'][1], dtype=torch.float32) for bi in range(bs)], dim=0).to(device)
            relative_prompt_batch_v_2 = torch.cat([torch.tensor(relative_train_prompts[train_data_idx_list_convert[bi]]['pred_ligand_v'][1]) for bi in range(bs)], dim=0).long().squeeze().to(device)
            # relative_prompt_batch_batch_2 = torch.repeat_interleave(torch.arange(bs), torch.tensor(relative_train_prompts['num_atom'])).to(device)
            relative_prompt_batch_batch_2_list = []
            for bi in range(bs):
                relative_prompt_batch_batch_2_list += [torch.tensor(bi)] * relative_train_prompts[train_data_idx_list_convert[bi]]['pred_ligand_v'][1].shape[0]
            relative_prompt_batch_batch_2 = torch.tensor(relative_prompt_batch_batch_2_list).long().to(device)

            relative_prompt_batch_pos_3 = torch.cat([torch.tensor(relative_train_prompts[train_data_idx_list_convert[bi]]['pred_ligand_pos'][2], dtype=torch.float32) for bi in range(bs)], dim=0).to(device)
            relative_prompt_batch_v_3 = torch.cat([torch.tensor(relative_train_prompts[train_data_idx_list_convert[bi]]['pred_ligand_v'][2]) for bi in range(bs)], dim=0).long().squeeze().to(device)
            # relative_prompt_batch_batch = torch.repeat_interleave(torch.arange(bs), torch.tensor(relative_train_prompts['num_atom'])).to(device)
            relative_prompt_batch_batch_3_list = []
            for bi in range(bs):
                relative_prompt_batch_batch_3_list += [torch.tensor(bi)] * relative_train_prompts[train_data_idx_list_convert[bi]]['pred_ligand_v'][2].shape[0]
            relative_prompt_batch_batch_3 = torch.tensor(relative_prompt_batch_batch_3_list).long().to(device)

            relative_prompt_batch = {'ligand_pos': relative_prompt_batch_pos, 'ligand_v': relative_prompt_batch_v,
                                     'ligand_batch': relative_prompt_batch_batch}
            relative_prompt_batch_2 = {'ligand_pos': relative_prompt_batch_pos_2, 'ligand_v': relative_prompt_batch_v_2,
                                       'ligand_batch': relative_prompt_batch_batch_2}
            relative_prompt_batch_3 = {'ligand_pos': relative_prompt_batch_pos_3, 'ligand_v': relative_prompt_batch_v_3,
                                       'ligand_batch': relative_prompt_batch_batch_3}

            # protein_noise = torch.randn_like(batch.protein_pos) * config.train.pos_noise_std
            # gt_protein_pos = batch.protein_pos + protein_noise
            gt_protein_pos = batch.protein_pos
            
            rel_pos = relative_prompt_batch['ligand_pos']
            rel_v = relative_prompt_batch['ligand_v']
            rel_batch = relative_prompt_batch['ligand_batch']

            # -------- start: diffusion loss --------
            results = model.get_diffusion_loss(
                normal_net_cond=normal_net_cond,
                relative_net_cond=relative_net_cond,

                protein_pos=gt_protein_pos,
                protein_v=batch.protein_atom_feature.float(),
                batch_protein=batch.protein_element_batch,

                ligand_pos=batch.ligand_pos,
                ligand_v=batch.ligand_atom_feature_full,
                batch_ligand=batch.ligand_element_batch,

                # normal
                normal_prompt_ligand_pos=normal_prompt_batch.ligand_pos,
                normal_prompt_ligand_v=normal_prompt_batch.ligand_atom_feature_full,
                normal_prompt_batch_ligand=normal_prompt_batch.ligand_element_batch,

                # relative
                relative_prompt_ligand_pos=rel_pos,
                relative_prompt_ligand_v=rel_v,
                relative_prompt_batch_ligand=rel_batch,

#                 relative_prompt_ligand_pos_2=relative_prompt_batch_2[
#                     'ligand_pos'] if relative_prompt_batch_2 is not None else None,
#                 relative_prompt_ligand_v_2=relative_prompt_batch_2[
#                     'ligand_v'] if relative_prompt_batch_2 is not None else None,
#                 relative_prompt_batch_ligand_2=relative_prompt_batch_2[
#                     'ligand_batch'] if relative_prompt_batch_2 is not None else None,

#                 relative_prompt_ligand_pos_3=relative_prompt_batch_3[
#                     'ligand_pos'] if relative_prompt_batch_3 is not None else None,
#                 relative_prompt_ligand_v_3=relative_prompt_batch_3[
#                     'ligand_v'] if relative_prompt_batch_3 is not None else None,
#                 relative_prompt_batch_ligand_3=relative_prompt_batch_3[
#                     'ligand_batch'] if relative_prompt_batch_3 is not None else None
            )
            loss, loss_pos, loss_v = results['loss'], results['loss_pos'], results['loss_v']
            loss = loss / config.train.n_acc_batch
            loss.backward()
        orig_grad_norm = clip_grad_norm_(model.parameters(), config.train.max_grad_norm)
        optimizer.step()

        if it % args.train_report_iter == 0:
            logger.info(
                '[Train] Iter %d | Loss %.6f (pos %.6f | v %.6f) | Lr: %.6f | Grad Norm: %.6f' % (
                    it, loss, loss_pos, loss_v, optimizer.param_groups[0]['lr'], orig_grad_norm
                )
            )
            # for k, v in results.items():
            #     if torch.is_tensor(v) and v.squeeze().ndim == 0:
            #         writer.add_scalar(f'train/{k}', v, it)
            # writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], it)
            # writer.add_scalar('train/grad', orig_grad_norm, it)
            # writer.flush()


    def validate(it):
        # fix time steps
        sum_loss, sum_loss_pos, sum_loss_v, sum_n = 0, 0, 0, 0
        sum_loss_bond, sum_loss_non_bond = 0, 0
        all_pred_v, all_true_v = [], []
        all_pred_bond_type, all_gt_bond_type = [], []
        with torch.no_grad():
            model.eval()
            for all_batch in tqdm(val_loader, desc='Validate'):
                all_batch = [b.to(args.device) for b in all_batch]
                assert len(all_batch) == normal_topk_prompt + 2, "wrong value of topk_prompt"

                normal_prompt_batch_2, normal_prompt_batch_3 = None, None
                if normal_topk_prompt == 1:
                    batch, normal_prompt_batch, test_data_idx = all_batch
                elif normal_topk_prompt == 2:
                    batch, normal_prompt_batch, normal_prompt_batch_2, test_data_idx = all_batch
                elif normal_topk_prompt == 3:
                    batch, normal_prompt_batch, normal_prompt_batch_2, normal_prompt_batch_3, test_data_idx = all_batch
                else:
                    raise ValueError(normal_topk_prompt)

                batch_size = batch.num_graphs


                # todo: test prompts
                device = args.device
                test_data_idx = test_data_idx.unsqueeze(0).to('cpu')
                while len(test_data_idx.shape) > 1:
                    test_data_idx = test_data_idx.squeeze(0)
                while len(test_data_idx.shape) == 0:
                    test_data_idx = test_data_idx.unsqueeze(0)
                test_data_idx_list = [test_data_idx[i].item() for i in range(test_data_idx.shape[0])]
                test_data_idx_list_convert = [relative_test_prompts_convert_index[i] for i in test_data_idx_list]
                val_bs = len(test_data_idx_list)

                # ['data', 'pred_ligand_pos', 'pred_ligand_v']
                val_relative_prompt_batch_pos = torch.cat([torch.tensor(
                    relative_train_prompts[test_data_idx_list_convert[bi]]['pred_ligand_pos'][0], dtype=torch.float32)
                    for bi in range(val_bs)], dim=0).to(device)
                val_relative_prompt_batch_v = torch.cat(
                    [torch.tensor(relative_train_prompts[test_data_idx_list_convert[bi]]['pred_ligand_v'][0]) for bi in
                     range(val_bs)], dim=0).long().squeeze().to(device)
                val_relative_prompt_batch_batch_1_list = []
                for bi in range(val_bs):
                    val_relative_prompt_batch_batch_1_list += [torch.tensor(bi)] * \
                                                          relative_train_prompts[test_data_idx_list_convert[bi]][
                                                              'pred_ligand_v'][0].shape[0]
                val_relative_prompt_batch_batch = torch.tensor(val_relative_prompt_batch_batch_1_list).long().to(device)

                val_relative_prompt_batch_pos_2 = torch.cat([torch.tensor(
                    relative_train_prompts[test_data_idx_list_convert[bi]]['pred_ligand_pos'][1], dtype=torch.float32)
                                                         for bi in range(val_bs)], dim=0).to(device)
                val_relative_prompt_batch_v_2 = torch.cat(
                    [torch.tensor(relative_train_prompts[test_data_idx_list_convert[bi]]['pred_ligand_v'][1]) for bi in
                     range(val_bs)], dim=0).long().squeeze().to(device)
                val_relative_prompt_batch_batch_2_list = []
                for bi in range(val_bs):
                    val_relative_prompt_batch_batch_2_list += [torch.tensor(bi)] * \
                                                          relative_train_prompts[test_data_idx_list_convert[bi]][
                                                              'pred_ligand_v'][1].shape[0]
                val_relative_prompt_batch_batch_2 = torch.tensor(val_relative_prompt_batch_batch_2_list).long().to(device)

                val_relative_prompt_batch_pos_3 = torch.cat([torch.tensor(
                    relative_train_prompts[test_data_idx_list_convert[bi]]['pred_ligand_pos'][2], dtype=torch.float32)
                                                         for bi in range(val_bs)], dim=0).to(device)
                val_relative_prompt_batch_v_3 = torch.cat(
                    [torch.tensor(relative_train_prompts[test_data_idx_list_convert[bi]]['pred_ligand_v'][2]) for bi in
                     range(val_bs)], dim=0).long().squeeze().to(device)
                val_relative_prompt_batch_batch_3_list = []
                for bi in range(val_bs):
                    val_relative_prompt_batch_batch_3_list += [torch.tensor(bi)] * \
                                                          relative_train_prompts[test_data_idx_list_convert[bi]][
                                                              'pred_ligand_v'][2].shape[0]
                val_relative_prompt_batch_batch_3 = torch.tensor(val_relative_prompt_batch_batch_3_list).long().to(device)

                val_relative_prompt_batch = {'ligand_pos': val_relative_prompt_batch_pos, 'ligand_v': val_relative_prompt_batch_v,
                                         'ligand_batch': val_relative_prompt_batch_batch}
                val_relative_prompt_batch_2 = {'ligand_pos': val_relative_prompt_batch_pos_2,
                                           'ligand_v': val_relative_prompt_batch_v_2,
                                           'ligand_batch': val_relative_prompt_batch_batch_2}
                val_relative_prompt_batch_3 = {'ligand_pos': val_relative_prompt_batch_pos_3,
                                           'ligand_v': val_relative_prompt_batch_v_3,
                                           'ligand_batch': val_relative_prompt_batch_batch_3}

                t_loss, t_loss_pos, t_loss_v = [], [], []
                for t in np.linspace(0, model.num_timesteps - 1, 10).astype(int):
                    time_step = torch.tensor([t] * batch_size).to(args.device)
                    results = model.get_diffusion_loss(
                        normal_net_cond=normal_net_cond,
                        relative_net_cond=relative_net_cond,
                        protein_pos=batch.protein_pos,
                        protein_v=batch.protein_atom_feature.float(),
                        batch_protein=batch.protein_element_batch,

                        ligand_pos=batch.ligand_pos,
                        ligand_v=batch.ligand_atom_feature_full,
                        batch_ligand=batch.ligand_element_batch,

                        # normal
                        normal_prompt_ligand_pos=normal_prompt_batch.ligand_pos,
                        normal_prompt_ligand_v=normal_prompt_batch.ligand_atom_feature_full,
                        normal_prompt_batch_ligand=normal_prompt_batch.ligand_element_batch,

                        # relative
                        relative_prompt_ligand_pos=batch.ligand_pos, ##val_relative_prompt_batch['ligand_pos'],
                        relative_prompt_ligand_v=batch.ligand_atom_feature_full,#val_relative_prompt_batch['ligand_v'],
                        relative_prompt_batch_ligand=batch.ligand_element_batch,#val_relative_prompt_batch['ligand_batch'],

#                         relative_prompt_ligand_pos_2=val_relative_prompt_batch_2[
#                             'ligand_pos'] if val_relative_prompt_batch_2 is not None else None,
#                         relative_prompt_ligand_v_2=val_relative_prompt_batch_2[
#                             'ligand_v'] if val_relative_prompt_batch_2 is not None else None,
#                         relative_prompt_batch_ligand_2=val_relative_prompt_batch_2[
#                             'ligand_batch'] if val_relative_prompt_batch_2 is not None else None,

#                         relative_prompt_ligand_pos_3=val_relative_prompt_batch_3[
#                             'ligand_pos'] if val_relative_prompt_batch_3 is not None else None,
#                         relative_prompt_ligand_v_3=val_relative_prompt_batch_3[
#                             'ligand_v'] if val_relative_prompt_batch_3 is not None else None,
#                         relative_prompt_batch_ligand_3=val_relative_prompt_batch_3[
#                             'ligand_batch'] if val_relative_prompt_batch_3 is not None else None,

                        time_step=time_step
                    )
                    loss, loss_pos, loss_v = results['loss'], results['loss_pos'], results['loss_v']

                    sum_loss += float(loss) * batch_size
                    sum_loss_pos += float(loss_pos) * batch_size
                    sum_loss_v += float(loss_v) * batch_size
                    sum_n += batch_size
                    all_pred_v.append(results['ligand_v_recon'].detach().cpu().numpy())
                    all_true_v.append(batch.ligand_atom_feature_full.detach().cpu().numpy())

        avg_loss = sum_loss / sum_n
        avg_loss_pos = sum_loss_pos / sum_n
        avg_loss_v = sum_loss_v / sum_n
        atom_auroc = get_auroc(np.concatenate(all_true_v), np.concatenate(all_pred_v, axis=0),
                               feat_mode=config.data.transform.ligand_atom_mode)

        if config.train.scheduler.type == 'plateau':
            scheduler.step(avg_loss)
        elif config.train.scheduler.type == 'warmup_plateau':
            scheduler.step_ReduceLROnPlateau(avg_loss)
        else:
            scheduler.step()

        logger.info(
            '[Validate] Iter %05d | Loss %.6f | Loss pos %.6f | Loss v %.6f e-3 | Avg atom auroc %.6f' % (
                it, avg_loss, avg_loss_pos, avg_loss_v * 1000, atom_auroc
            )
        )
        # writer.add_scalar('val/loss', avg_loss, it)
        # writer.add_scalar('val/loss_pos', avg_loss_pos, it)
        # writer.add_scalar('val/loss_v', avg_loss_v, it)
        # writer.flush()
        return avg_loss


    try:
        best_loss, best_iter = None, None
        for it in range(1, config.train.max_iters + 1):
            # with torch.autograd.detect_anomaly():
            train(it)
            if it % config.train.val_freq == 0 or it == config.train.max_iters:
                val_loss = validate(it)
                if best_loss is None or val_loss < best_loss:
                    logger.info(f'[Validate] Best val loss achieved: {val_loss:.6f}')
                    best_loss, best_iter = val_loss, it
                    ckpt_path = os.path.join(ckpt_dir, '%d.pt' % it)
                    torch.save({
                        'config': config,
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'iteration': it,
                    }, ckpt_path)
                else:
                    logger.info(f'[Validate] Val loss is not improved. '
                                f'Best val loss: {best_loss:.6f} at iter {best_iter}')
    except KeyboardInterrupt:
        logger.info('Terminating...')

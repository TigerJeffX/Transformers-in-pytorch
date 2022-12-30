# encoding=utf8
import os
import torch
import torchsummary
import sys

sys.path.append('../')

from data.nmt_task import SOS, EOS, PAD, MAX_LEN
from data.nmt_task import get_Multi30K_data_loader
from model.vanilla_transformer import VanillaTransformer, MASK, Loss, NEG_INF
from torchtext.data.metrics import bleu_score as cal_bleu_score
from ml_collections import ConfigDict
from tqdm import tqdm

import torch.multiprocessing as mp
import torch.distributed as dist
from apex.parallel import DistributedDataParallel as DDP

def get_args(dataset):
    args = ConfigDict()
    if dataset == 'multi30k':
        args.embedding_dim = 256
        args.max_len = MAX_LEN
        args.hidden_dim = 256
        args.head_num = 8
        args.enc_layer_num = 2
        args.dec_layer_num = 2
        args.smoothing_confidence = 0.7
        args.beam_width = 3
        args.decoder_alpha = 0.9
        args.world_size = 4
        args.epoch = 15
        args.use_fp16 = False
        args.val_freq = 20
        args.ckp_after = 300
        args.train_batch_size = 512//4
        args.test_batch_size = 128
        args.share_emb = True
        args.warmup_steps = 200
    else:
        raise ValueError("dataset not in [multi30k] but %s"%dataset)
    return args

def get_model(args, src_vocab, trg_vocab):
    enc_num_embeddings = len(src_vocab.stoi)
    dec_num_embeddings = len(trg_vocab.stoi)
    model = VanillaTransformer(
        enc_num_embeddings,
        dec_num_embeddings,
        args.max_len,
        args.embedding_dim,
        args.hidden_dim,
        args.head_num,
        args.hidden_dim*4,
        args.enc_layer_num,
        args.dec_layer_num,
        args.share_emb,
    )
    return model

def idx_to_tok(t, src_vocab, trg_vocab, src_or_trg='trg'):
    ''' idx convert to token by lookup vocab
    '''
    term_results = []
    idx_results = t.cpu().detach().numpy()
    for r in range(0, idx_results.shape[0]):
        term_result = []
        for c in range(0, idx_results.shape[1]):
            idx = idx_results[r][c]
            if src_or_trg == 'trg':
                tok = trg_vocab.itos[idx]
                term_result.append(tok)
                if tok == EOS:
                    break
            elif src_or_trg == 'src':
                tok = src_vocab.itos[idx]
                term_result.append(tok)
                if tok == EOS:
                    break
            else:
                raise ValueError("term result must be in ['trg', 'src'] but %s"%src_or_trg)
        term_results.append(term_result)
    return term_results

def tok_to_bleu_format(src_input_t, trg_label_t, trg_pred_t, src_vocab, trg_vocab, show_cases=False):
    ''' convert token for calculate bleu score
    '''
    src_list = idx_to_tok(src_input_t, src_vocab, trg_vocab, 'src')
    trg_label_list = idx_to_tok(trg_label_t, src_vocab, trg_vocab, 'trg')
    trg_pred_list = idx_to_tok(trg_pred_t, src_vocab, trg_vocab, 'trg')
    candidate_corpus = []
    references_corpus = []
    cnt = 1
    for src, trg_label, trg_pred in zip(src_list, trg_label_list, trg_pred_list):
        if show_cases and cnt<=2:
            print('src:', src)
            print('trg_label:', trg_label)
            print('trg_pred:', trg_pred)
            print('='*80)
        candidate_corpus.append(trg_pred)
        references_corpus.append([trg_label])
        cnt += 1
    return candidate_corpus, references_corpus

def decorate_batch(batch, enc_padding_idx, dec_padding_idx, device_idx):
    src = batch['src'].transpose(0, 1).cuda(device_idx)
    trg = batch['trg'].transpose(0, 1).cuda(device_idx)
    enc_x, dec_x, dec_y = src, trg[:, :-1], trg[:, 1:]
    padding_method = 1
    enc_self_mask = MASK.create_mask(
        enc_x, enc_x, enc_padding_idx, enc_padding_idx, device_idx,
        use_causal_mask=False, padding_method=padding_method)
    dec_self_mask = MASK.create_mask(
        dec_x, dec_x, dec_padding_idx, dec_padding_idx, device_idx,
        use_causal_mask=True, padding_method=padding_method)
    dec_cross_mask = MASK.create_mask(
        dec_x, enc_x, dec_padding_idx, enc_padding_idx, device_idx,
        use_causal_mask=False, padding_method=padding_method)
    ret = {
        'enc_x':enc_x,
        'dec_x':dec_x,
        'dec_y':dec_y,
        'enc_self_mask':enc_self_mask,
        'dec_self_mask':dec_self_mask,
        'dec_cross_mask':dec_cross_mask
    }
    return ret

def inference_batch(
        args,
        ddp_model,
        batch,
        enc_padding_idx,
        dec_padding_idx,
        src_vocab,
        trg_vocab,
        device_idx,
        infer_type):

    src = batch['src'].transpose(0, 1)
    trg = batch['trg'].transpose(0, 1)
    enc_x, dec_y = src, trg[:, 1:]
    padding_method = 1
    enc_self_mask = MASK.create_mask(
        enc_x, enc_x, enc_padding_idx, enc_padding_idx, device_idx,
        use_causal_mask=False, padding_method=padding_method
    )
    o_enc = ddp_model.module.encoder(enc_x, enc_self_mask)

    def _greedy_search():
        infer_y = torch.empty(enc_x.size(0), 1).fill_(trg_vocab.stoi[SOS]).type_as(enc_x.data)
        EOS_slots = torch.BoolTensor(enc_x.size(0)).fill_(False).cuda(device_idx)
        for i in range(args.max_len-1):
            dec_self_mask = MASK.create_mask(
                infer_y, infer_y, dec_padding_idx, dec_padding_idx, device_idx,
                use_causal_mask=True, padding_method=padding_method
            )
            dec_cross_mask = MASK.create_mask(
                infer_y, enc_x, dec_padding_idx, enc_padding_idx, device_idx,
                use_causal_mask=False, padding_method=padding_method
            )
            o_dec = ddp_model.module.decoder(o_enc, infer_y, dec_self_mask, dec_cross_mask)
            o_out = ddp_model.module.output(o_dec)
            pred_next_tok = o_out.max(-1).indices.type_as(infer_y.data)[:,-1:]
            infer_y = torch.cat([infer_y, pred_next_tok], dim=-1)
            EOS_slots = EOS_slots | (pred_next_tok.squeeze()==trg_vocab.stoi[EOS])
            if EOS_slots.sum().item() >= EOS_slots.size(0):
                break
        return infer_y[:,1:]

    def _beam_search(beam_width=3):
        hist_infer_y = torch.empty(enc_x.size(0), 1, 1).fill_(trg_vocab.stoi[SOS]).type_as(enc_x.data)
        hist_prob_y = torch.empty(enc_x.size(0), 1, 1).fill_(1.).type_as(o_enc.data)
        EOS_slots = torch.BoolTensor(enc_x.size(0), beam_width).fill_(False).cuda(device_idx)
        decoder_max_len = int(enc_x.size(1)*1.4)
        for i in range(decoder_max_len-1):
            step_candidate_infer_y = []
            step_candidate_infer_prob = []
            for b in range(hist_infer_y.size(-1)):
                input_y = hist_infer_y[:,:,b]
                prob_y = hist_prob_y[:,:,b]
                dec_self_mask = MASK.create_mask(
                    input_y, input_y, dec_padding_idx, dec_padding_idx, device_idx,
                    use_causal_mask=True, padding_method=padding_method
                )
                dec_cross_mask = MASK.create_mask(
                    input_y, enc_x, dec_padding_idx, enc_padding_idx, device_idx,
                    use_causal_mask=False, padding_method=padding_method
                )
                o_dec = ddp_model.module.decoder(o_enc, input_y, dec_self_mask, dec_cross_mask)
                o_out_latest = ddp_model.module.output(o_dec)[:,-1]
                next_beam_search_y = o_out_latest.topk(k=beam_width, dim=-1).indices.type_as(input_y.data)
                next_beam_search_prob = o_out_latest.topk(k=beam_width, dim=-1).values
                next_beam_search_y.masked_fill_(input_y[:,-1:] == trg_vocab.stoi[EOS],  trg_vocab.stoi[EOS])
                next_beam_search_prob[:,0].masked_fill_(input_y[:,-1] == trg_vocab.stoi[EOS], 0.)
                next_beam_search_prob[:,1:].masked_fill_(input_y[:,-1:] == trg_vocab.stoi[EOS], NEG_INF)
                candidate_infer_y = torch.cat(
                    [
                        input_y.unsqueeze(2).expand(input_y.size(0), input_y.size(1), beam_width),
                        next_beam_search_y.unsqueeze(1)
                    ],
                    dim=1
                )
                # candidate_infer_y → [batch_size, seq_len, beam_width]
                candidate_infer_prob = torch.cat(
                    [
                        prob_y.unsqueeze(2).expand(prob_y.size(0), prob_y.size(1), beam_width),
                        next_beam_search_prob.unsqueeze(1)
                    ],
                    dim=1
                )
                step_candidate_infer_y.append(candidate_infer_y)
                step_candidate_infer_prob.append(candidate_infer_prob)

            seq_len = (torch.cat(step_candidate_infer_y, dim=-1)!=trg_vocab.stoi[EOS]).sum(dim=1)
            seq_len = (seq_len+1).pow(args.decoder_alpha)
            seq_sum = torch.cat(step_candidate_infer_prob, dim=-1).sum(dim=1)
            step_beam_search_idx = seq_sum.div(seq_len).topk(k=beam_width, dim=-1).indices.type_as(input_y.data)
            hist_infer_y = torch.cat(step_candidate_infer_y, dim=-1)
            hist_prob_y = torch.cat(step_candidate_infer_prob, dim=-1)
            hist_infer_y = hist_infer_y.gather(
                2,
                step_beam_search_idx.unsqueeze(1).expand(
                    hist_infer_y.size(0),
                    hist_infer_y.size(1),
                    step_beam_search_idx.size(1)
                )
            )
            hist_prob_y = hist_prob_y.gather(
                2,
                step_beam_search_idx.unsqueeze(1).expand(
                    hist_prob_y.size(0),
                    hist_prob_y.size(1),
                    step_beam_search_idx.size(1)
                )
            )
            # beam search stop condition
            EOS_slots = EOS_slots | (hist_infer_y[:,-1,:]==trg_vocab.stoi[EOS])
            eos_hits = (EOS_slots>0).sum().item()
            if eos_hits >= EOS_slots.size(0)*EOS_slots.size(0):
                break
        return hist_infer_y[:,1:,:]

    if infer_type=='greedy_search':
        pred_y = _greedy_search()
        return enc_x, dec_y, pred_y
    elif infer_type=='beam_search':
        pred_y_all_beam = _beam_search(args.beam_width)
        return enc_x, dec_y, pred_y_all_beam
    else:
        raise ValueError('infer type must in ["greedy_search","beam_search"] but %s'%infer_type)

def do_test(args, ddp_model, checkpoint_path, test_loader, enc_padding_idx, dec_padding_idx, src_vocab, trg_vocab, cret, rank):
    def _test_greedy(model):
        model.eval()
        candidate_corpus = []
        references_corpus = []
        for step, test_batch in enumerate(test_loader):
            enc_x, dec_y, infer_y = inference_batch(
                args, model, test_batch, enc_padding_idx, dec_padding_idx, src_vocab, trg_vocab, rank, infer_type='greedy_search')
            candidate_corpus_batch, references_corpus_batch = \
                tok_to_bleu_format(enc_x, dec_y, infer_y, src_vocab, trg_vocab, show_cases=False)
            candidate_corpus += candidate_corpus_batch
            references_corpus += references_corpus_batch
        total_bleu_score = cal_bleu_score(candidate_corpus, references_corpus)*100
        print("---Infer type : %s"%('greedy_search'))
        print("---Test nmt pair : %s"%(len(candidate_corpus)))
        print("---Test BLEU score in greedy search : %s"%('{:.4f}'.format(total_bleu_score)))

    def _test_beam(model):
        model.eval()
        candidate_corpus = [[]for _ in range(args.beam_width)]
        references_corpus = []
        for step, test_batch in enumerate(test_loader):
            enc_x, dec_y, infer_y = inference_batch(
                args, model, test_batch, enc_padding_idx, dec_padding_idx, src_vocab, trg_vocab, rank, infer_type='beam_search')
            candidate_corpus_batch, references_corpus_batch = \
                tok_to_bleu_format(enc_x, dec_y, infer_y[:,:,0], src_vocab, trg_vocab, show_cases=False)
            candidate_corpus[0] += candidate_corpus_batch
            references_corpus += references_corpus_batch
            for b in range(1, args.beam_width):
                candidate_corpus_batch, _ = \
                    tok_to_bleu_format(enc_x, dec_y, infer_y[:,:,b], src_vocab, trg_vocab, show_cases=False)
                candidate_corpus[b] += candidate_corpus_batch
        print("---Infer type : %s"%('beam_search'))
        print("---Test nmt pair : %s"%(len(candidate_corpus[0])))
        for b in range(args.beam_width):
            total_bleu_score = cal_bleu_score(candidate_corpus[b], references_corpus)*100
            print("---Test BLEU score in beam search %s : %s"%(b, '{:.4f}'.format(total_bleu_score)))

    map_location = {'cuda:%d'%rank:'cuda:%d'%rank}
    ddp_model.load_state_dict(torch.load(checkpoint_path, map_location=map_location))
    print('Reloaded model parameter from %s'%checkpoint_path)
    print('Conduct Testing...')
    with torch.no_grad():
        _test_beam(ddp_model)
        _test_greedy(ddp_model)

def do_eval(model, eval_data_loader, enc_padding_idx, dec_padding_idx, src_vocab, trg_vocab, cret, rank):
    eval_progress = tqdm(
        eval_data_loader,
        bar_format="{l_bar}{r_bar}",
        dynamic_ncols=True,
        disable=rank not in [0]
    )
    model.eval()
    candidate_corpus = []
    references_corpus = []
    eval_loss = 0.
    eval_non_padding_sum = 0
    for step, batch in enumerate(eval_progress):
        with torch.no_grad():
            b = decorate_batch(
                batch=batch, enc_padding_idx=enc_padding_idx, dec_padding_idx=dec_padding_idx, device_idx=rank)
            o_model = model.forward(
                b['enc_x'], b['dec_x'], b['enc_self_mask'], b['dec_self_mask'], b['dec_cross_mask'])
            loss, non_padding_sum = cret(o_model, b['dec_y'])
            eval_loss += loss.item()
        # update eval progress
        eval_non_padding_sum += non_padding_sum
        avg_eval_loss = eval_loss/eval_non_padding_sum
        eval_progress.set_description("Valid (Loss=%2.4f) "%(avg_eval_loss))
        # generate translation result
        candidate_corpus_batch, references_corpus_batch = \
            tok_to_bleu_format(b['enc_x'], b['dec_y'], o_model.max(-1).indices, src_vocab, trg_vocab)
        candidate_corpus += candidate_corpus_batch
        references_corpus += references_corpus_batch
    # record eval result
    avg_eval_loss = eval_loss/eval_non_padding_sum
    eval_bleu_score = cal_bleu_score(candidate_corpus, references_corpus)*100
    return avg_eval_loss, eval_bleu_score

def do_train(device_id, args):

    rank = device_id

    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=args.world_size,
        rank=rank
    )

    torch.cuda.set_device(device_id)

    # 1. Data & Vocab
    train_loader, val_loader, test_loader, src_vocab, trg_vocab = get_Multi30K_data_loader(args, device_id)
    if device_id==0:
        print("src vocab size: %s" % len(src_vocab.stoi))
        print("tgt vocab size: %s" % len(trg_vocab.stoi))
    enc_padding_idx = src_vocab[PAD]
    dec_padding_idx = trg_vocab[PAD]

    # 2. Model
    model = get_model(args, src_vocab, trg_vocab).cuda(device_id)
    if device_id==0:
        torchsummary.summary(model)
    ddp_model = DDP(model, gradient_predivide_factor=args.world_size)

    # 3. Loss
    cret = Loss(
        padding_idx=dec_padding_idx,
        smoothing_confidence=args.smoothing_confidence)
    cret = cret.cuda(device_id)

    # 4. Opt
    opt = torch.optim.Adam(model.parameters(), lr=0.5, betas=(0.9, 0.98), eps=1e-9)
    def _rate(step, model_size, factor, warmup):
        if step == 0:
            step = 1
        return factor * (model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5)))
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer = opt,
        lr_lambda = lambda step: _rate(step, model_size=args.hidden_dim, factor=1.0, warmup=400)
    )

    # 5. Train & Valid
    total_step = 0
    best_bleu_score = 0.0
    best_step = total_step
    checkpoint_path = 'vanilla_transformer_0.checkpoint'
    for epoch in range(args.epoch):
        train_epoch_progress = tqdm(
            train_loader, bar_format="{l_bar}{r_bar}", dynamic_ncols=True, disable=rank not in [0])
        train_loader.sampler.set_epoch(epoch)
        train_sample_total = 0
        train_loss_total = 0
        for step, batch in enumerate(train_epoch_progress):
            ddp_model.train()
            # decorate batch data (create mask)
            b = decorate_batch(
                batch=batch, enc_padding_idx=enc_padding_idx, dec_padding_idx=dec_padding_idx, device_idx=device_id)
            # forward
            o_model = ddp_model.forward(
                b['enc_x'], b['dec_x'], b['enc_self_mask'], b['dec_self_mask'], b['dec_cross_mask'])
            # loss & backward
            loss, non_padding_sum = cret(o_model, b['dec_y'])
            loss.backward()
            # lr & optimizer
            opt.step()
            opt.zero_grad()
            lr_scheduler.step()
            # update epoch train progress info
            train_sample_total += b['enc_x'].shape[0]
            train_loss_total += loss.item()
            avg_epoch_loss = train_loss_total/(train_sample_total*args.world_size)
            last_lr = lr_scheduler.get_last_lr()[0]
            train_epoch_progress.set_description(
                "Train (Epoch=%s, Step=%s, Loss=%2.4f, lr=%1.6f) "%(epoch, step, avg_epoch_loss, last_lr))
            total_step += 1
            dist.barrier()
            if total_step>=args.ckp_after and total_step%args.val_freq==0 and rank==0:
                avg_eval_loss, eval_bleu_score = \
                    do_eval(model, val_loader, enc_padding_idx, dec_padding_idx, src_vocab, trg_vocab, cret, rank)
                print('eval bleu %s at step %s'%(eval_bleu_score, total_step))
                if eval_bleu_score > best_bleu_score:
                    best_bleu_score = eval_bleu_score
                    best_step = total_step
                    print('best eval bleu %s at step %s'%(best_bleu_score, best_step))
                    checkpoint_path = 'vanilla_transformer_%s.checkpoint'%total_step
                    torch.save(ddp_model.state_dict(), checkpoint_path)

    dist.barrier()
    # 6. Test
    if rank==0:
        do_test(
            args, ddp_model, checkpoint_path, test_loader, enc_padding_idx, dec_padding_idx, src_vocab, trg_vocab, cret, rank)

if __name__ == '__main__':

    dataset = sys.argv[1]

    args = get_args(dataset)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "8096"

    mp.spawn(do_train, nprocs=args.world_size, args=(args,))

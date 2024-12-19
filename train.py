import os 

import time
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter

import pandas as pd
from sklearn.model_selection import train_test_split

from config import hp
from dataset import BaseDataset, text_mel_collate_fn
from tts_loss import TTSLoss
from model import TransformerTTS
from transform import inverse_mel_spec_to_wav, text_to_seq


def batch_process(batch):
    text_padded, \
    text_lengths, \
    mel_padded, \
    mel_lengths, \
    stop_token_padded = batch

    text_padded = text_padded.cuda()
    text_lengths = text_lengths.cuda()
    mel_padded = mel_padded.cuda()
    stop_token_padded = stop_token_padded.cuda()
    mel_lengths = mel_lengths.cuda()

    N = mel_padded.shape[0]
    SOS = torch.zeros((N, 1, hp.mel_freq), device=mel_padded.device)

    mel_input = torch.cat(
        [
          SOS,
          mel_padded[:, :-1, :]
        ],
        dim=1
    )

    return text_padded, \
         text_lengths, \
         mel_padded, \
         mel_lengths, \
         mel_input, \
         stop_token_padded



def inference(model, text):
    sequences = text_to_seq(text).unsqueeze(0).cuda()
    postnet_mel, _ = model.inference(
        sequences,
        stop_token_threshold=1e5,
        with_tqdm = False
    )
    audio = inverse_mel_spec_to_wav(postnet_mel.detach()[0].T)

    return audio


def calculate_test_loss(model, test_loader, criterion):
    test_loss_mean = 0.0
    model.eval()

    with torch.no_grad():
        for test_i, test_batch in enumerate(test_loader):
            test_text_padded, \
            test_text_lengths, \
            test_mel_padded, \
            test_mel_lengths, \
            test_mel_input, \
            test_stop_token_padded = batch_process(test_batch)

            test_post_mel_out, test_mel_out, test_stop_token_out = model(
                test_text_padded,
                test_text_lengths,
                test_mel_input,
                test_mel_lengths
            )
            test_loss = criterion(
                mel_postnet_out = test_post_mel_out,
                mel_out = test_mel_out,
                stop_token_out = test_stop_token_out,
                mel_target = test_mel_padded,
                stop_token_target = test_stop_token_padded
            )

            test_loss_mean += test_loss.item()

            test_loss_mean = test_loss_mean / (test_i + 1)
    return test_loss_mean


def main():
    torch.manual_seed(hp.seed)
    df = pd.read_csv(hp.csv_path)  
    train_df, test_df = train_test_split(df, test_size=128, random_state=hp.seed)
    train_loader = torch.utils.data.DataLoader(
        BaseDataset(train_df),
        num_workers=8,
        shuffle=True,
        sampler=None,
        batch_size=64,
        pin_memory=True,
        drop_last=True,
        collate_fn=text_mel_collate_fn,
    )
    test_loader = torch.utils.data.DataLoader(
        BaseDataset(test_df),
        num_workers=8,
        shuffle=True,
        sampler=None,
        batch_size=64,
        pin_memory=True,
        drop_last=True,
        collate_fn=text_mel_collate_fn,
    )

    test_saved_path = f"{hp.save_path}/{hp.save_name}"
    logger = SummaryWriter(hp.log_path)
    criterion = TTSLoss().cuda()
    model = TransformerTTS().cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=hp.lr)
    scaler = torch.cuda.amp.GradScaler()

    best_test_loss_mean = float("inf")

    train_loss_mean = 0.0
    epoch = 0
    i = 0

    while True:
        progress_bar = tqdm(
            train_loader,
            desc=f'Training, epoch:{epoch + 1},'
        )
        for j, batch in enumerate(progress_bar):
            text_padded, \
            text_lengths, \
            mel_padded, \
            mel_lengths, \
            mel_input, \
            stop_token_padded = batch_process(batch)

            model.train(True)
            model.zero_grad()

            with torch.autocast(device_type='cuda', dtype=torch.float16):
                post_mel_out, mel_out, stop_token_out = model(
                    text_padded, 
                    text_lengths,
                    mel_input,
                    mel_lengths
                )
                loss = criterion(
                  mel_postnet_out=post_mel_out,
                  mel_out = mel_out,
                  stop_token_out = stop_token_out,
                  mel_target = mel_padded,
                  stop_token_target = stop_token_padded
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), hp.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            train_loss_mean += loss.item()      
            i += 1

        ## evaluate
        test_loss_mean = calculate_test_loss(model, test_loader, criterion)
        audio = inference(model, "Hello, World.")

        logger.add_scalar("Loss/test_loss", test_loss_mean, global_step=i) 
        logger.add_audio(f"Utterance/audio_{i}",audio, sample_rate=hp.sr, global_step=i)

        print(f"{epoch}-{i}) Test loss: {np.round(test_loss_mean, 5)}")

        train_loss_mean = train_loss_mean / j
        logger.add_scalar("Loss/train_loss", train_loss_mean, global_step=i)

        is_best_test = test_loss_mean < best_test_loss_mean

        state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "i": i,
            "test_loss": test_loss_mean,
            "train_loss": train_loss_mean
        }

        if is_best_test:
            torch.save(state, test_saved_path)
            best_test_loss_mean = test_loss_mean


        print(f"{epoch}-{i}) Train loss: {np.round(train_loss_mean, 5)}")
        train_loss_mean = 0.0
        epoch += 1

if __name__ == "__main__":
    main()

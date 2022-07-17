import argparse
import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import PIL.Image as pil_image
import os

from model import Net
from preprocess import convert_rgb_to_ycbcr, convert_ycbcr_to_rgb
from metrics import calc_psnr


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights-file', type=str, required=True)
    parser.add_argument('--video-file', type=str, required=True)
    parser.add_argument('--bicubic-frames', type=str, required=True)
    parser.add_argument('--vsr-frames', type=str, required=True)
    parser.add_argument('--vid-name', type=str, default='video')
    parser.add_argument('--scale', type=int, default=3)
    args = parser.parse_args()

    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = torch.load(args.weights_file)
    model.eval()

    vid = cv2.VideoCapture(args.video_file)
    frNo = 0
    avg_psnr = 0
    avg_bi_psnr = 0
    while(True):
        success, frame = vid.read()
        if success:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = pil_image.fromarray(frame, 'RGB')

            image_width = (image.width // args.scale) * args.scale
            image_height = (image.height // args.scale) * args.scale
            image = image.resize((image_width, image_height), resample=pil_image.Resampling.BICUBIC)
            OG = torch.from_numpy(convert_rgb_to_ycbcr(np.array(image).astype(np.float32))[..., 0]/255.).to(device).unsqueeze(0).unsqueeze(0)
            image = image.resize((image.width // args.scale, image.height // args.scale), resample=pil_image.Resampling.BICUBIC)
            image = image.resize((image.width * args.scale, image.height * args.scale), resample=pil_image.Resampling.BICUBIC)
            image.save(args.bicubic_frames + 'Frame_{}.jpg'.format(frNo))

            image = np.array(image).astype(np.float32)
            ycbcr = convert_rgb_to_ycbcr(image)

            y = ycbcr[..., 0]
            y /= 255.
            y = torch.from_numpy(y).to(device)
            y = y.unsqueeze(0).unsqueeze(0)

            with torch.no_grad():
                preds = model(OG).clamp(0.0, 1.0)

            bi_psnr = calc_psnr(OG, y)
            psnr = calc_psnr(OG, preds)
            # print('PSNR: {:.2f}'.format(psnr))

            preds = preds.mul(255.0).cpu().numpy().squeeze(0).squeeze(0)

            output = np.array([preds, ycbcr[..., 1], ycbcr[..., 2]]).transpose([1, 2, 0])
            output = np.clip(convert_ycbcr_to_rgb(output), 0.0, 255.0).astype(np.uint8)
            output = pil_image.fromarray(output)
            output.save(args.vsr_frames + f'frame_{frNo}.jpg')

            avg_psnr += psnr
            avg_bi_psnr += bi_psnr
        else:
            break
        frNo += 1

    vid.release()

    sr_images = os.listdir(args.vsr_frames)
    sr_images.sort(key=len)
    bi_images = os.listdir(args.bicubic_frames)
    bi_images.sort(key=len)

    sr_height, sr_width, channels = cv2.imread('C:/Users/girishgodage/PycharmProjects/assignment-1-0-JackFrost-0/' + args.vsr_frames + sr_images[0]).shape
    height, width, _ = cv2.imread('C:/Users/girishgodage/PycharmProjects/assignment-1-0-JackFrost-0/' + args.bicubic_frames + bi_images[0]).shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    new_vid = cv2.VideoWriter(args.vid_name+f'_vsr_x{args.scale}.avi', fourcc, 31, (sr_width, sr_height))

    for filename in sr_images:
        img = cv2.imread('C:/Users/girishgodage/PycharmProjects/assignment-1-0-JackFrost-0/' + args.vsr_frames + filename)
        new_vid.write(img)
    new_vid.release()

    bi_vid = cv2.VideoWriter(args.vid_name + f'_bicubic_x{args.scale}.avi', fourcc, 31, (width, height))

    for filename in bi_images:
        img = cv2.imread('C:/Users/girishgodage/PycharmProjects/assignment-1-0-JackFrost-0/' + args.bicubic_frames + filename)
        bi_vid.write(img)
    bi_vid.release()

    # avg_bi_psnr /= (frNo+1)
    avg_psnr /= (frNo+1)
    print('The psnr of SR with original is:', avg_psnr)
    print('The psnr of bicubic with original is:', avg_bi_psnr)

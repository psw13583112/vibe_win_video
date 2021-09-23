from multi_person_tracker import MPT
import torch
import cv2
from torchvision.transforms.functional import to_tensor
import numpy as np

from utils import (
    convert_crop_cam_to_orig_img,
    prepare_rendering_results,
    get_single_image_crop_demo,
)
from utils import Renderer


def prepare_output_tracks(trackers):
    '''
    Put results into a dictionary consists of detected people
    :param trackers (ndarray): input tracklets of shape Nx5 [x1,y1,x2,y2,track_id]
    :return: dict: of people. each key represent single person with detected bboxes and frame_ids
    '''
    people = dict()
    bbox_list = []
    for frame_idx, tracks in enumerate(trackers):
        for d in tracks:
            person_id = int(d[4])
            # bbox = np.array([d[0], d[1], d[2] - d[0], d[3] - d[1]]) # x1, y1, w, h

            w, h = d[2] - d[0], d[3] - d[1]
            c_x, c_y = d[0] + w / 2, d[1] + h / 2
            w = h = np.where(w / h > 1, w, h)
            bbox = np.array([c_x, c_y, w, h])

            if person_id in people.keys():
                people[person_id]['bbox'].append(bbox)
                people[person_id]['frames'].append(frame_idx)
            else:
                people[person_id] = {
                    'bbox': [],
                    'frames': [],
                }
                people[person_id]['bbox'].append(bbox)
                people[person_id]['frames'].append(frame_idx)
            bbox_list.append(bbox)
    for k in people.keys():
        people[k]['bbox'] = np.array(people[k]['bbox']).reshape((len(people[k]['bbox']), 4))
        people[k]['frames'] = np.array(people[k]['frames'])

    return bbox_list


def main():
    device = 'cuda'
    # model = VIBE_Demo(
    #     seqlen=16,
    #     n_layers=2,
    #     hidden_size=1024,
    #     add_linear=True,
    #     use_residual=True,
    # ).to(device)
    model = torch.jit.load("3d.pt").to(device)
    mot = MPT(
        device=device,
        batch_size=1,
        display=False,
        detector_type="yolo",
        output_format='dict',
        yolo_img_size=416,
    )
    # image_folder = "./tmp/test_mp4/000001.png"
    # img_ = cv2.imread(image_folder)
    # orig_height,orig_width = img.shape[:2]
    # img = cv2.cvtColor(img_, cv2.COLOR_BGR2RGB)
    # input = to_tensor(img).reshape([1,3,orig_height,orig_width]).cuda().float()
    capture = cv2.VideoCapture(0)
    orig_width,orig_height = capture.get(3),capture.get(4)
    # pretrained_file = download_ckpt(use_3dpw=False)
    # ckpt = torch.load(pretrained_file)
    # ckpt = ckpt['gen_state_dict']
    # model.load_state_dict(ckpt, strict=False)
    model.eval()

    with torch.no_grad():
        while True:
            ret, img_ = capture.read()
            if not ret :
                break
            if ret == True:
                img = cv2.cvtColor(img_, cv2.COLOR_BGR2RGB)
                input_im = to_tensor(img).unsqueeze(0)
                tracking_results = mot.run_tracker_one(input_im)
                result = prepare_output_tracks(tracking_results)
                if not result:
                    cv2.imshow("1", img)
                    if cv2.waitKey(1) == ord('q'):
                        break
                    continue
                # img = to_tensor(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                norm_img, raw_img, kp_2d = get_single_image_crop_demo(
                    img,
                    result[0],
                    kp_2d=None,
                    scale=1.0,
                    crop_size=224)
                pred_cam, pred_verts, pred_pose, pred_betas, pred_joints3d, norm_joints2d = [], [], [], [], [], []
                batch = norm_img.unsqueeze(0).unsqueeze(0)
                batch = batch.to(device)

                batch_size, seqlen = batch.shape[:2]
                output = model(batch)[-1]
                pred_cam.append(output[0][:, :, :3].reshape(batch_size * seqlen, -1))
                pred_verts.append(output[1].reshape(batch_size * seqlen, -1, 3))
                pred_pose.append(output[0][:, :, 3:75].reshape(batch_size * seqlen, -1))
                pred_betas.append(output[0][:, :, 75:].reshape(batch_size * seqlen, -1))
                pred_joints3d.append(output[3].reshape(batch_size * seqlen, -1, 3))
                pred_cam = torch.cat(pred_cam, dim=0)
                pred_verts = torch.cat(pred_verts, dim=0)
                pred_pose = torch.cat(pred_pose, dim=0)
                pred_betas = torch.cat(pred_betas, dim=0)
                pred_joints3d = torch.cat(pred_joints3d, dim=0)
                pred_cam = pred_cam.cpu().detach().numpy()
                pred_verts = pred_verts.cpu().detach().numpy()
                pred_pose = pred_pose.cpu().detach().numpy()
                pred_betas = pred_betas.cpu().detach().numpy()
                pred_joints3d = pred_joints3d.cpu().detach().numpy()

                orig_cam = convert_crop_cam_to_orig_img(
                    cam=pred_cam,
                    bbox=np.array(result),
                    img_width=orig_width,
                    img_height=orig_height
                )
                output_dict = {
                    'pred_cam': pred_cam,
                    'orig_cam': orig_cam,
                    'verts': pred_verts,
                    'pose': pred_pose,
                    'betas': pred_betas,
                    'joints3d': pred_joints3d,
                    'joints2d': None,
                    'bboxes': result,
                    'frame_ids': [0,],
                }
                vibe_results = {}
                vibe_results[0] = output_dict
                renderer = Renderer(resolution=(orig_width, orig_height), orig_img=True, wireframe=False)
                frame_results = prepare_rendering_results(vibe_results, 1)
                print(frame_results[0][0])
                frame_verts = frame_results[0][0]['verts']
                frame_cam = frame_results[0][0]['cam']
                img_result = renderer.render(
                    img_,
                    frame_verts,
                    cam=frame_cam,
                    color=(1.0, 0.5, 0.6401947423208725),
                    mesh_filename=None,
                )
                cv2.imshow("1",img_result)
                if cv2.waitKey(1) == ord('q'):
                    break


if __name__ == '__main__':
    main()
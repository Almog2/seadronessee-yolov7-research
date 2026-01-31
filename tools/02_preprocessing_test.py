import os

if __name__ == "__main__":
    cmd = (
        "python3 test.py "
        "--data data/seadronessee_tiled_train.yaml "
        "--img 640 "
        "--weights runs/train/seadronessee_tiled_train/weights/best.pt "
        "--task val "
        "--device 0 "
        "--name seadronessee_tiled_test "
        "--conf-thres 0.25 "
        "--iou-thres 0.5 "
        "--save-conf "
        "--save-txt "
    )

    print("Running TEST command:")
    print(cmd)
    os.system(cmd)

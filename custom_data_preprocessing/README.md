# StableVITON Custom Data Preprocessing

HR-VITON [<Issue#45>](https://github.com/sangyun884/HR-VITON/issues/45) 참고해서 만듦.

* OpenPose
* Human Parse
* DensePose
* Agnostic Map

이렇게 4종의 데이터 전처리를 해줍니다. 이 4개만 있으면 StableVITON inference 가능.

# Dependencies and Environments
* `envs/` 에 있는 yaml로 conda 환경 3개 생성

* "hand" 추정 가능한 OpenPose 설치
    * 설치 후 `bash/run_openpose.sh`에서 `openpose_path` 변경 필요
    
* `Self-Correction-Human-Parsing/pretrain_model`에 [exp-schp-201908261155-lip.pth](https://drive.google.com/file/d/1k4dllHpu0bdx38J7H28rVVLpU-kOHmnH/view?usp=sharing) 다운로드

# Usage
1. 768 * 1024 사이즈로 사람 이미지 준비하여 `data/image/`에 넣기
1. `custom_data_preprocessing` 폴더에서 이 순서대로 실행
```
. bash/run_openpose.sh
. bash/run_densepose.sh
. bash/run_human_parsing.sh
conda run -n agnostic_map python script/agnostic_map.py
```
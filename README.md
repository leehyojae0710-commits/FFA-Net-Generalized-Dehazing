# FFA-Net 기반 이미지 안개 제거 연구

**합성·현실 안개 데이터셋의 조합을 통한 범용 이미지 안개 제거 모델 학습 및 평가**

이 저장소는 FFA-Net(Feature Fusion Attention Network)을 기반으로 안개 제거(Image Dehazing) 모델의 **학습 데이터 구성 전략**을 연구한 프로젝트의 소스코드입니다. RESIDE(ITS/OTS), Haze4K, NH-HAZE 등 서로 다른 도메인의 데이터를 이용해 모델을 학습하고, PSNR·SSIM 및 시각적 결과를 비교할 수 있도록 구성했습니다.

> **논문 게재 완료:** 이 프로젝트와 관련된 논문은 2026년 9월 30일 *Journal of Multimedia Information System (JMIS)*에 게재되었습니다. 아래에 공식 논문 링크와 정확한 인용 정보를 기재했습니다.

## 1. 게재 논문

| 항목 | 내용 |
| --- | --- |
| 제목 | **Data Composition Strategy for FFA-Net Image Dehazing** |
| 저자 | Chae-bin Park, Hyo-jae Lee, Jae-Ho Nah |
| 학술지 | *Journal of Multimedia Information System (JMIS)* |
| 발행 정보 | 2026, Vol. 13, No. 3, pp. 91–100 |
| 온라인 게재일 | 2026-09-30 |
| DOI | [10.33851/JMIS.2026.13.3.91](https://doi.org/10.33851/JMIS.2026.13.3.91) |
| 논문 원문 | [JMIS 공식 논문 페이지](https://www.jmis.org/archive/view_article?pid=jmis-13-3-91) |

### 연구 목적 및 주요 결과

- 서로 다른 안개 환경에서 하나의 모델을 안정적으로 활용할 수 있도록 **학습 데이터 조합 전략**을 분석했습니다.
- 실내·실외 합성 데이터(ITS/OTS)와 현실 비균일 안개 데이터(NH-HAZE), Haze4K를 활용해 서로 다른 조건의 모델을 비교했습니다.
- 게재 논문에서는 **Haze4K와 NH-HAZE를 1:1로 혼합한 학습 전략**이 다양한 환경에서 전반적으로 안정적인 결과를 제공하여 권장된다고 보고합니다. 10:1 구성도 유효한 대안으로 분석했습니다.
- 평가에는 PSNR, SSIM 및 시각적 복원 결과를 활용했습니다.

> 위 결과는 **게재 논문에 보고된 연구 결과**입니다. 저장소에 포함된 Python 파일과 학습 설정만으로 모든 결과가 자동 재현되었다고 주장하는 것은 아닙니다. 모델 가중치 및 원본 데이터셋은 별도 준비가 필요합니다.

## 2. 프로젝트 구성 및 주요 기능

| 구성 요소 | 설명 |
| --- | --- |
| 모델 | PyTorch 기반 FFA-Net 구현 및 Perceptual Loss |
| 학습 | 데이터셋 선택, 학습·검증, 체크포인트 저장 및 재학습 |
| 평가 | PSNR/SSIM 기반 성능 평가, 이미지 추론 결과 저장 |
| 데이터 처리 | ITS, OTS, NH-HAZE4K 등 데이터 경로 처리 및 데이터 증강 |
| GUI | Tkinter 기반 이미지 선택·모델 로드·안개 제거 결과 비교 |
| 실험 | 단일 데이터셋, 증강 데이터셋, 혼합 데이터셋 구성 비교 |

## 3. 프로젝트 디렉터리

~~~text
FFA-Net-Generalized-Dehazing/
├── README.md
├── requirements.txt
├── .gitignore
├── GITHUB_UPLOAD_GUIDE.md
└── net/
    ├── __init__.py
    ├── main.py                 # 모델 학습
    ├── test.py                 # 모델 추론 및 결과 저장
    ├── gui.py                  # Tkinter GUI
    ├── option.py               # 학습 옵션
    ├── data_utils.py           # 데이터셋 로더
    ├── metrics.py              # PSNR / SSIM
    ├── models/
    │   ├── __init__.py
    │   ├── FFA.py              # FFA-Net 모델
    │   └── PerceptualLoss.py
    └── trained_models/
        └── readme.md           # 가중치 안내
~~~

**GitHub에 포함하지 않은 자료**

- `data/`: 대용량 학습·평가 데이터셋
- `net/trained_models/*.pk`: 학습 완료 모델 가중치 7개
- `net/test_imgs/`, `net/test_imgs(clear)/`: 실험용 입력 및 정답 이미지
- 연구 논문 PDF: 학술지 원문 링크로 대체
- `.git/`, `__pycache__/`: 원본 Git 이력과 자동 생성 파일

원본 프로젝트 자료: [Google Drive의 FFA-Net 폴더](https://drive.google.com/drive/folders/1guRC-IiGevj-Za4jYkoYWJPhKtWs908V)

## 4. 권장 실행 환경

- Python 3.8 이상
- PyTorch / torchvision
- CUDA 지원 GPU 권장 (CPU에서도 일부 기능은 동작할 수 있으나 속도 차이가 큼)
- Windows 또는 Linux
- 주요 패키지: numpy, matplotlib, opencv-python, Pillow, scikit-image, tqdm

~~~bash
git clone https://github.com/leehyojae0710-commits/FFA-Net-Generalized-Dehazing.git
cd FFA-Net-Generalized-Dehazing
pip install -r requirements.txt
~~~

`requirements.txt`는 소스코드의 import와 기존 README를 기준으로 정리한 패키지 목록입니다. **특정 Python/CUDA/PyTorch 버전 조합에서의 실행은 별도로 검증해야 합니다.**

## 5. 데이터셋 준비

학습 코드는 다음 디렉터리 구조를 가정합니다. 데이터셋은 GitHub에 포함되지 않으므로 원 제공처의 이용 조건에 따라 내려받아 배치해야 합니다.

~~~text
FFA-Net-Generalized-Dehazing/
├── data/
│   └── RESIDE/
│       ├── ITS/
│       │   ├── train/
│       │   │   ├── hazy/
│       │   │   └── clear/
│       │   └── test/
│       │       ├── hazy/
│       │       └── clear/
│       ├── OTS/
│       │   ├── train/
│       │   │   ├── hazy/
│       │   │   └── clear/
│       │   └── test/
│       │       ├── hazy/
│       │       └── clear/
│       └── NH-HAZE4K/
│           ├── train/
│           │   ├── hazy/
│           │   └── clear/
│           └── test/
│               ├── hazy/
│               └── clear/
└── net/
    ├── test_imgs/
    └── trained_models/
~~~

데이터셋 원본 출처:

- **RESIDE:** https://sites.google.com/view/reside-dehaze-datasets/reside-standard
- **Haze4K:** https://github.com/liuye123321/DMT-Net
- **NH-HAZE:** https://data.vision.ee.ethz.ch/cvl/ntire20/nh-haze/

## 6. 학습 명령어

다음 명령어는 **`net/` 폴더로 이동한 상태**에서 실행합니다.

~~~bash
cd net
~~~

### ITS 학습

~~~bash
python main.py --net=ffa --crop --crop_size=240 --blocks=19 --gps=3 --bs=2 --lr=0.0001 --trainset=its_train --testset=its_test --steps=500000 --eval_step=50000
~~~

### OTS 학습

~~~bash
python main.py --net=ffa --crop --crop_size=240 --blocks=19 --gps=3 --bs=2 --lr=0.0001 --trainset=ots_train --testset=ots_test --steps=500000 --eval_step=50000
~~~

### NH-HAZE4K 학습

~~~bash
python main.py --net=ffa --crop --crop_size=240 --blocks=19 --gps=3 --bs=2 --lr=0.0001 --trainset=nh4k_train --testset=nh4k_test --steps=500000 --eval_step=50000
~~~

학습 과정에서 평가 주기에 따라 체크포인트가 `net/trained_models/`에 저장됩니다. 실험 재현을 위해 실제 사용 데이터, 모델 파일, 학습 옵션을 맞춰야 합니다.

## 7. 테스트 및 GUI 실행

### 명령줄 테스트

~~~bash
# ITS용 체크포인트
python test.py --task=its --test_imgs=test_imgs

# OTS용 체크포인트
python test.py --task=ots --test_imgs=test_imgs

# NH-HAZE4K용 체크포인트
python test.py --task=nh4k --test_imgs=test_imgs

# trained_models 폴더에 있는 모든 .pk 파일 순회
python test.py --test_imgs=test_imgs
~~~

**주의:** `--task` 옵션을 사용하는 경우 코드가 정해진 이름의 모델을 찾습니다. 예를 들어 `--task=nh4k`는 `nh4k_train_ffa_3_19.pk`를 요구하므로, `(10by1)` 등의 접미사가 붙은 체크포인트는 이름 또는 로딩 로직을 맞춰야 합니다.

### GUI 실행

~~~bash
python gui.py
~~~

GUI에서 `*.pk` 모델 파일과 입력 이미지를 선택하면 안개 제거 전후 이미지를 비교할 수 있습니다. **공개 저장소에는 모델 가중치가 없으므로**, GUI 추론 전 별도로 준비해야 합니다.

## 8. 학습 모델(별도 보관)

원본 프로젝트 폴더에서 확인한 체크포인트 목록입니다. **아래 파일은 GitHub 저장소에 포함되지 않습니다.**

| 실험 구성 | 파일명 |
| --- | --- |
| RESIDE ITS | `its_train_ffa_3_19.pk` |
| RESIDE OTS | `ots_train_ffa_3_19.pk` |
| Haze4K | `haze4k_train_ffa_3_19.pk` |
| NH-HAZE | `nh-haze_train_ffa_3_19.pk` |
| NH-HAZE 증강 | `nh-haze_train_ffa_3_19(aug).pk` |
| NH-HAZE + Haze4K 혼합 (5:1 파일 표기) | `nh4k_train_ffa_3_19(5by1).pk` |
| NH-HAZE + Haze4K 혼합 (10:1 파일 표기) | `nh4k_train_ffa_3_19(10by1).pk` |

혼합 비율 및 파일명은 기존 프로젝트 자료의 표기를 따릅니다. **논문에서 권장한 1:1 결과**와 이 폴더의 체크포인트 목록을 동일한 것으로 간주해서는 안 됩니다.

## 9. 논문 인용 및 참고 문헌

본 연구를 인용할 때는 아래의 **실제 출판된 논문**을 사용해 주세요.

~~~bibtex
@article{Park2026DataComposition,
  title   = {Data Composition Strategy for FFA-Net Image Dehazing},
  author  = {Park, Chae-bin and Lee, Hyo-jae and Nah, Jae-Ho},
  journal = {Journal of Multimedia Information System},
  year    = {2026},
  volume  = {13},
  number  = {3},
  pages   = {91--100},
  doi     = {10.33851/JMIS.2026.13.3.91}
}
~~~

기반 연구 및 코드 출처:

- **FFA-Net 원 논문:** X. Qin et al., “FFA-Net: Feature Fusion Attention Network for Single Image Dehazing,” *AAAI*, 2020.
- **기반 코드 저장소:** https://github.com/zhilin007/FFA-Net
- **본 연구 논문:** https://www.jmis.org/archive/view_article?pid=jmis-13-3-91

## 10. 라이선스 및 공개 범위

이 프로젝트는 기존 FFA-Net 구현을 기반으로 합니다. 원본 README에 있던 **GNU Affero General Public License(AGPL) v3-or-later 고지**를 아래에 유지했습니다. 제3자 코드·데이터셋의 재배포 조건은 각각 확인해야 합니다. **학술지 논문의 CC BY-NC 4.0 이용 조건과 소스코드의 이용 조건은 서로 별개**입니다.

원본 README의 고지:

> Copyright (c) 2026 Author(s)
>
> This program is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
>
> This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more details.
>
> You should have received a copy of the GNU Affero General Public License along with this program. If not, see <http://www.gnu.org/licenses/>.

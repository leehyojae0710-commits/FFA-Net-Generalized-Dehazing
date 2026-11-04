# GitHub 업로드 안내 / Publishing notes

## 이 패키지의 범위

Google Drive `포트폴리오/FFA-Net` 폴더에서 Python 소스코드와 기존 README, gitignore, 모델 안내문을 가져와 GitHub 저장소에 업로드하기 편하도록 정리한 파일입니다. **GitHub에 실제로 게시된 상태는 아닙니다.** 원본 코드의 기능을 변경하지 않았으며, 실행 테스트를 완료한 패키지라는 뜻도 아닙니다.

원본 폴더: https://drive.google.com/drive/folders/1guRC-IiGevj-Za4jYkoYWJPhKtWs908V
기반 연구/원본 코드: https://github.com/zhilin007/FFA-Net

## 포함

- `README.md`: Google Drive의 원본 프로젝트 설명
- `net/main.py`, `net/data_utils.py`, `net/option.py`, `net/metrics.py`: 학습 및 데이터 처리 코드
- `net/test.py`, `net/gui.py`: 평가 및 Tkinter GUI 코드
- `net/models/`: 네트워크 및 loss 구현 코드
- `.gitignore`, `requirements.txt`: GitHub 업로드/의존성 관리 보조 파일
- `net/trained_models/readme.md`: 체크포인트 다운로드 관련 안내

## GitHub에서 제외한 파일

- `.git/`: 원본 저장소의 내부 이력; 새 GitHub 저장소로 가져가지 않음
- `data/`: RESIDE 등 대용량 학습·평가 데이터셋; 원본 제공 조건 확인 후 별도 다운로드 권장
- `net/trained_models/*.pk`: GPU 학습 체크포인트 7개, 개당 약 18~27 MB; 배포 시 Git LFS 또는 별도 파일 공유 권장
- `net/test_imgs/`, `net/test_imgs(clear)/`: 원본 데이터 출처·재배포 권한 확인 전 공개 제외
- `A Generalized Training Method for Dehazing Networks.pdf`: 약 44 MB 연구 PDF. 공동저작/제출 상태와 공개 가능 여부 확인 후 별도 게시 권장
- `__pycache__/`: 생성 캐시

## 실행 시 주의

원본 README의 명령어는 일반적으로 `net/` 디렉터리에서 실행하는 형태입니다. 학습 데이터와 `.pk` 파일을 동봉하지 않았으므로 ZIP 파일만 받아서는 모델 학습/추론을 바로 실행할 수 없습니다.

```bash
cd net
python gui.py   # 준비된 .pk 파일을 직접 선택하여 GUI 추론
# 또는 원본 README에 적힌 main.py / test.py 명령어 사용
```

원본 README에는 Python 3.8 이상 환경과 CUDA GPU가 권장되어 있으며 `requirements.txt`는 코드 import와 README에서 파악되는 패키지 이름만 모아둔 것입니다. 특정 버전 조합에서 실행 가능하다는 보증은 아닙니다.

## GitHub에 새 저장소로 업로드하기

1. GitHub 계정에서 새 **빈** 저장소 생성 (추천 이름: `FFA-Net-Generalized-Dehazing`).
2. 이 ZIP을 먼저 **압축 해제**한 뒤, 내부 프로젝트 폴더의 파일을 사용.
3. 새 저장소의 **Add file → Upload files**로 파일 및 하위 폴더 업로드, Commit.
4. 업로드 직후 README의 실행 절차·참조 링크 및 GitHub 파일 목록 점검.

기존 원본 코드(`zhilin007/FFA-Net`)를 기반으로 수정한 프로젝트이므로 원본 프로젝트의 출처를 반드시 유지해야 합니다. 코드의 재배포 및 라이선스 선택 가능 여부는 원저작자 조건을 별도로 확인하세요. 현재 README의 라이선스 문구가 원본 프로젝트의 모든 코드에 적용되는지 확인되지 않았으므로 새로운 라이선스 파일은 임의로 생성하지 않았습니다.

## 개인 정보 / 보안

공개 전 `git status`, `.gitignore`, 비밀번호, 토큰, API 키, 로컬 경로가 들어간 비공개 설정을 추가 확인하세요. 제공된 코드에 대해 별도의 실행 테스트나 외부 네트워크 테스트는 수행하지 않았습니다.
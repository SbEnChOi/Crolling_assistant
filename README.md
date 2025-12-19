<img width="2547" height="1332" alt="image" src="https://github.com/user-attachments/assets/113d5f04-19d0-444b-9281-58c01ba6672c" />

# 크롤링 도우미

대상 매체에서 에서 키워드로 게시물을 크롤링하고 CSV 파일로 저장하는 프로그램입니다.
문장을 입력하면 자동으로 키워드를 추출하고 영어로 번역하여 검색합니다.

## 주요 기능

- **자동 키워드 추출**: 문장에서 키워드를 자동으로 추출 (형태소 분석, 불용어 제거)
- **영어 번역**: 한국어 키워드를 영어로 자동 번역하여 Reddit 검색 최적화
- **다중 키워드 검색**: 여러 키워드로 동시 검색하여 더 많은 데이터 수집
- **날짜 필터링**: 지정한 날짜 범위 내의 게시물만 수집
- **CSV 저장**: 수집된 데이터를 DataFrame과 CSV 파일로 저장

## 설치 방법

### 1. 필요한 라이브러리 설치

```bash
pip install -r requirements.txt
```

**참고**: KoNLPy는 Java가 필요합니다. Windows에서는 [JDK 설치](https://www.oracle.com/java/technologies/downloads/)가 필요할 수 있음.

### 2. Reddit API 키 발급

1. https://www.reddit.com/prefs/apps 접속
2. "create app" 또는 "create another app" 클릭
3. 앱 타입: "script" 선택
4. 앱 이름, 설명, redirect uri 입력 (redirect uri는 http://localhost:8080 등 임의로 설정)
5. 생성 후 다음 정보 확인:
   - Client ID (앱 이름 아래에 있는 문자열)
   - Client Secret (secret 부분)

### 3. .env 파일 생성

프로젝트 루트 디렉토리에 `.env` 파일을 생성하고 다음 내용을 입력:

```
REDDIT_CLIENT_ID=your_client_id_here
REDDIT_CLIENT_SECRET=your_client_secret_here
REDDIT_USER_AGENT=your_app_name/1.0 by your_username
```

`REDDIT_USER_AGENT`는 "앱이름/버전 by 레딧유저네임" 형식으로 작성하세요.

## 사용 방법

```bash
python main.py
```

프로그램 실행 후 다음 정보를 입력:

1. **크롤링할 매체**: 현재는 'reddit'만 지원
2. **검색할 내용**: 크롤링하고 싶은 내용을 담은 문장 입력
   - 예: "파이썬 프로그래밍 학습 방법"
   - 프로그램이 자동으로 키워드를 추출하고 표시합니다
3. **키워드 선택**: 추출된 키워드 중 사용할 키워드를 선택
   - 키워드를 콤마로 구분하여 직접 입력 (예: 파이썬,프로그래밍,학습)
   - 'all' 입력 시 전체 키워드 사용
   - 입력한 키워드가 추출된 목록에 없으면 경고 메시지 표시
4. **시작 날짜**: YYYY-MM-DD 형식
5. **끝 날짜**: YYYY-MM-DD 형식
6. **크롤링할 데이터 개수**: 수집할 게시물 개수

## 키워드 추출 과정

1. **형태소 분석**: KoNLPy의 Okt를 사용하여 문장을 분석
2. **품사 추출**: 명사, 동사, 형용사만 추출
3. **불용어 제거**: 조사, 접속사 등 불필요한 단어 제거
4. **사용자 선택**: 추출된 키워드 중 사용자가 원하는 키워드 선택
5. **영어 번역**: 선택된 한국어 키워드를 영어로 번역
6. **검색 실행**: 한국어 키워드와 영어 키워드 모두로 검색

## 결과

- **CSV 파일**: 현재 디렉토리에 저장됩니다
- **파일명 형식**: `reddit_{키워드}_{시작날짜}_{끝날짜}.csv`
- **수집되는 데이터**:
  - 제목 (title)
  - 본문 내용 (content)
  - 작성자 (author)
  - 업보트 수 (score)
  - 댓글 수 (num_comments)
  - URL (url)
  - 작성 시간 (created_date)
  - 서브레딧 이름 (subreddit)
  - 검색 키워드 (search_keyword): 어떤 키워드로 찾았는지 기록

## 주의사항

- Reddit API는 rate limit이 있습니다 (분당 약 60회 요청)
- 키워드가 많을수록 검색 시간이 오래 걸릴 수 있습니다
- 번역 서비스는 인터넷 연결이 필요합니다
- KoNLPy는 Java가 설치되어 있어야 정상 작동합니다



# 2025/11/16 
snscrape 로 twiiter 를 크롤링하는 것을 불가능해짐. 
즉 twitter 는 api 로 하는 것이 맞음

그 다음 reddit 은 reddit api 를 요청했지만 4일쨰 아무 소식이 없음 
snscrape 로 시도했지만 오류, 아마도 버전문제 또는 reddit 도 안되는 것일수도 있음

그럼 지금 할 수 있는 것은 남은 naver,youtube 를 크롤링하는 것.


# 2025/11/19 
일단 대상매체는 youtube 만 하기로 함. 
readme를 수정할 필요가 있음
reddit 은 정말 나쁜놈들임
이제 남은 과제는 파이썬 3.13 환경에서 모두 굴러가야함. 
그리고 꼭 필요한 파일 
client_secret.json
key.env
youtube_token.pickle 를 성수쌤한테 보내야함. 

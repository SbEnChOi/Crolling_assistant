"""
YouTube 크롤링 프로그램
YouTube Data API v3를 사용하여 동영상 및 댓글 데이터를 수집합니다.
"""

import os
import re
import pickle
from datetime import datetime
from dotenv import load_dotenv
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import pandas as pd
from keyword_extractor import extract_and_translate_keywords
from data_analysis import analyze_youtube_data

# 환경변수 로드
# key.env 파일을 명시적으로 로드
load_dotenv('key.env')
# .env 파일도 시도 (없어도 무방)
load_dotenv('.env')

# OAuth 2.0 설정
SCOPES = ['https://www.googleapis.com/auth/youtube.force-ssl']
CLIENT_SECRETS_FILE = 'client_secret.json'
TOKEN_FILE = 'youtube_token.pickle'

def sanitize_filename(text):
    """파일명에서 특수문자 제거 및 공백을 언더스코어로 변경"""
    text = re.sub(r'[^\w\s-]', '', text)
    text = re.sub(r'[-\s]+', '_', text)
    return text

def get_user_input():
    """콘솔에서 사용자 입력 받기"""
    print("=" * 50)
    print("YouTube 크롤링 프로그램")
    print("=" * 50)
    
    # 매체 선택 (현재는 youtube만)
    media = input("크롤링할 매체를 입력하세요 (현재는 'youtube'만 지원): ").strip().lower()
    if media != 'youtube':
        print("현재는 'youtube'만 지원합니다. 'youtube'으로 설정합니다.")
        media = 'youtube'
    
    # 검색할 내용 입력 (문장)
    search_sentence = input("크롤링하고 싶은 내용을 담은 문장을 입력하세요: ").strip()
    if not search_sentence:
        raise ValueError("검색할 내용을 입력해주세요.")
    
    # 키워드 추출 및 번역
    print("\n키워드 추출 중...")
    korean_keywords, english_keywords = extract_and_translate_keywords(search_sentence)
    
    # 모든 키워드 합치기 (한국어 + 영어)
    all_keywords = list(set(korean_keywords + english_keywords))
    
    print(f"\n선택된 키워드:")
    print(f"한국어: {', '.join(korean_keywords)}")
    print(f"영어: {', '.join(english_keywords)}")
    print(f"총 {len(all_keywords)}개의 키워드로 검색합니다.")
    
    # 시작 날짜
    start_date_str = input("시작 날짜를 입력하세요 (YYYY-MM-DD): ").strip()
    try:
        start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    except ValueError:
        raise ValueError("날짜 형식이 올바르지 않습니다. YYYY-MM-DD 형식으로 입력해주세요.")
    
    # 끝 날짜
    end_date_str = input("끝 날짜를 입력하세요 (YYYY-MM-DD): ").strip()
    try:
        end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
    except ValueError:
        raise ValueError("날짜 형식이 올바르지 않습니다. YYYY-MM-DD 형식으로 입력해주세요.")
    
    if start_date > end_date:
        raise ValueError("시작 날짜가 끝 날짜보다 늦을 수 없습니다.")
    
    # 크롤링할 데이터 개수
    try:
        max_videos = int(input("크롤링할 동영상 개수를 입력하세요: ").strip())
        if max_videos <= 0:
            raise ValueError("동영상 개수는 1 이상이어야 합니다.")
    except ValueError as e:
        if "invalid literal" in str(e):
            raise ValueError("동영상 개수는 숫자로 입력해주세요.")
        raise
    
    # 인증 방식 고정: OAuth 2.0 사용 (사용자에게 묻지 않음)
    #print("\n인증 방식: OAuth 2.0으로 고정")
    use_oauth = True
    
    # 댓글 수집 여부
    collect_comments = input("댓글도 수집하시겠습니까? (y/n, 기본값: n): ").strip().lower()
    collect_comments = collect_comments == 'y' or collect_comments == 'yes'
    
    # 정렬 기준
    print("\n정렬 기준을 선택하세요:")
    print("1. 관련도 (기본값)")
    print("2. 업로드 날짜 (최신순)")
    print("3. 조회수")
    print("4. 평점")
    
    sort_choice = input("선택 (1-4, 기본값: 1): ").strip()
    sort_map = {
        '1': 'relevance',
        '2': 'date',
        '3': 'viewCount',
        '4': 'rating'
    }
    order = sort_map.get(sort_choice, 'relevance')
    
    return media, all_keywords, korean_keywords, english_keywords, start_date, end_date, max_videos, collect_comments, order, use_oauth

def authenticate_oauth():
    """
    OAuth 2.0 인증을 통해 YouTube API 연결
    
    Returns:
        YouTube API 클라이언트 객체
    """
    credentials = None
    
    # 저장된 토큰이 있는지 확인
    if os.path.exists(TOKEN_FILE):
        try:
            with open(TOKEN_FILE, 'rb') as token:
                credentials = pickle.load(token)
        except Exception as e:
            print(f"저장된 토큰 로드 실패: {str(e)}")
    
    # 토큰이 없거나 만료된 경우 새로 인증
    if not credentials or not credentials.valid:
        if credentials and credentials.expired and credentials.refresh_token:
            # 토큰 갱신 시도
            try:
                print("토큰 갱신 중...")
                credentials.refresh(Request())
            except Exception as e:
                print(f"토큰 갱신 실패: {str(e)}")
                credentials = None
        
        if not credentials:
            # 새로운 인증 시작
            if not os.path.exists(CLIENT_SECRETS_FILE):
                raise FileNotFoundError(
                    f"{CLIENT_SECRETS_FILE} 파일을 찾을 수 없습니다.\n"
                    "Google Cloud Console에서 OAuth 2.0 클라이언트 ID를 생성하고 "
                    "JSON 파일을 다운로드하여 client_secret.json으로 저장하세요."
                )
            
            print("\nOAuth 2.0 인증을 시작합니다...")
            print("브라우저가 열리면 Google 계정으로 로그인하고 권한을 부여하세요.\n")
            
            flow = InstalledAppFlow.from_client_secrets_file(
                CLIENT_SECRETS_FILE, SCOPES)
            credentials = flow.run_local_server(port=0)
            
            print("\n인증이 완료되었습니다!")
        
        # 토큰 저장 (다음 실행 시 재사용)
        try:
            with open(TOKEN_FILE, 'wb') as token:
                pickle.dump(credentials, token)
            print(f"인증 토큰이 {TOKEN_FILE}에 저장되었습니다.")
        except Exception as e:
            print(f"토큰 저장 실패: {str(e)}")
    
    youtube = build('youtube', 'v3', credentials=credentials)
    return youtube


def connect_youtube_api_key():
    """API 키를 사용하여 YouTube Data API v3 연결"""
    try:
        api_key = os.getenv('YOUTUBE_API_KEY')
        
        if not api_key:
            # 디버깅 정보 출력
            print("\n[디버깅] 환경변수 확인:")
            print(f"  YOUTUBE_API_KEY: {api_key}")
            print(f"  key.env 파일 존재: {os.path.exists('key.env')}")
            print(f"  .env 파일 존재: {os.path.exists('.env')}")
            raise ValueError("환경변수에 YouTube API 키가 없습니다. key.env 또는 .env 파일에 YOUTUBE_API_KEY를 설정해주세요.")
        
        # API 키 앞뒤 공백 제거
        api_key = api_key.strip()
        
        youtube = build('youtube', 'v3', developerKey=api_key)
        return youtube
    except Exception as e:
        raise ConnectionError(f"YouTube API 연결에 실패했습니다: {str(e)}")


def connect_youtube(use_oauth=True):
    """
    YouTube Data API v3 연결 (OAuth 또는 API 키)
    
    Args:
        use_oauth: True면 OAuth 사용, False면 API 키 사용
    
    Returns:
        YouTube API 클라이언트 객체
    """
    if use_oauth:
        try:
            return authenticate_oauth()
        except Exception as e:
            print(f"\nOAuth 인증 실패: {str(e)}")
            print("API 키 방식으로 전환합니다...")
            return connect_youtube_api_key()
    else:
        return connect_youtube_api_key()

def search_videos(youtube, keyword, start_date, end_date, order='relevance', max_results=50):
    """
    키워드로 동영상 검색
    
    Args:
        youtube: YouTube API 클라이언트
        keyword: 검색 키워드
        start_date: 시작 날짜
        end_date: 끝 날짜
        order: 정렬 기준 (relevance, date, viewCount, rating)
        max_results: 최대 결과 개수
    
    Returns:
        동영상 ID 리스트
    """
    try:
        # 날짜 형식을 RFC 3339 형식으로 변환
        published_after = start_date.strftime('%Y-%m-%dT%H:%M:%SZ')
        published_before = (end_date.replace(hour=23, minute=59, second=59)).strftime('%Y-%m-%dT%H:%M:%SZ')
        
        # 검색 요청
        search_response = youtube.search().list(
            q=keyword,
            part='id,snippet',
            type='video',
            order=order,
            publishedAfter=published_after,
            publishedBefore=published_before,
            maxResults=min(max_results, 50),  # API 제한: 최대 50개
            videoDefinition='any',
            videoDuration='any',
            relevanceLanguage='en'  # 영어 콘텐츠 우선
        ).execute()
        
        video_ids = []
        for item in search_response.get('items', []):
            video_id = item['id']['videoId']
            video_ids.append(video_id)
        
        return video_ids
    
    except HttpError as e:
        print(f"  검색 오류: {str(e)}")
        return []

def get_video_details(youtube, video_ids):
    """
    동영상 상세 정보 수집
    
    Args:
        youtube: YouTube API 클라이언트
        video_ids: 동영상 ID 리스트
    
    Returns:
        동영상 정보 딕셔너리 리스트
    """
    try:
        # API는 한 번에 최대 50개까지만 조회 가능
        videos_data = []
        
        for i in range(0, len(video_ids), 50):
            batch_ids = video_ids[i:i+50]
            
            videos_response = youtube.videos().list(
                part='snippet,statistics,contentDetails',
                id=','.join(batch_ids)
            ).execute()
            
            for item in videos_response.get('items', []):
                snippet = item.get('snippet', {})
                statistics = item.get('statistics', {})
                content_details = item.get('contentDetails', {})
                
                # 업로드 날짜 파싱
                published_at = snippet.get('publishedAt', '')
                try:
                    upload_date = datetime.strptime(published_at[:10], '%Y-%m-%d')
                except:
                    upload_date = None
                
                video_data = {
                    'video_id': item.get('id', ''),
                    'title': snippet.get('title', ''),
                    'description': snippet.get('description', ''),
                    'url': f"https://www.youtube.com/watch?v={item.get('id', '')}",
                    'upload_date': upload_date.strftime('%Y-%m-%d') if upload_date else '',
                    'channel_name': snippet.get('channelTitle', ''),
                    'channel_id': snippet.get('channelId', ''),
                    'view_count': int(statistics.get('viewCount', 0)) if statistics.get('viewCount') else 0,
                    'like_count': int(statistics.get('likeCount', 0)) if statistics.get('likeCount') else 0,
                    'comment_count': int(statistics.get('commentCount', 0)) if statistics.get('commentCount') else 0,
                    'duration': content_details.get('duration', ''),
                    'thumbnail_url': snippet.get('thumbnails', {}).get('high', {}).get('url', ''),
                    'tags': ', '.join(snippet.get('tags', [])) if snippet.get('tags') else '',
                    'category_id': snippet.get('categoryId', '')
                }
                videos_data.append(video_data)
        
        return videos_data
    
    except HttpError as e:
        print(f"  동영상 정보 조회 오류: {str(e)}")
        return []

def get_video_comments(youtube, video_id, max_comments=100):
    """
    동영상 댓글 수집
    
    Args:
        youtube: YouTube API 클라이언트
        video_id: 동영상 ID
        max_comments: 최대 댓글 개수
    
    Returns:
        댓글 리스트
    """
    try:
        comments = []
        next_page_token = None
        
        while len(comments) < max_comments:
            try:
                results = youtube.commentThreads().list(
                    part='snippet',
                    videoId=video_id,
                    maxResults=min(100, max_comments - len(comments)),  # API 제한: 최대 100개
                    pageToken=next_page_token,
                    textFormat='plainText'
                ).execute()
                
                for item in results.get('items', []):
                    comment = item['snippet']['topLevelComment']['snippet']
                    comments.append({
                        'video_id': video_id,
                        'comment_id': item['snippet']['topLevelComment']['id'],
                        'author': comment.get('authorDisplayName', ''),
                        'text': comment.get('textDisplay', ''),
                        'like_count': int(comment.get('likeCount', 0)),
                        'published_at': comment.get('publishedAt', ''),
                        'updated_at': comment.get('updatedAt', '')
                    })
                
                next_page_token = results.get('nextPageToken')
                if not next_page_token:
                    break
                    
            except HttpError as e:
                if e.resp.status == 403:
                    # 댓글 비활성화된 동영상
                    break
                else:
                    print(f"    댓글 수집 오류 (video_id: {video_id}): {str(e)}")
                    break
        
        return comments
    
    except Exception as e:
        print(f"    댓글 수집 오류: {str(e)}")
        return []

def crawl_youtube(youtube, keywords, start_date, end_date, max_videos, collect_comments=False, order='relevance'):
    """
    YouTube에서 동영상 및 댓글 크롤링
    
    Args:
        youtube: YouTube API 클라이언트
        keywords: 검색할 키워드 리스트
        start_date: 시작 날짜
        end_date: 끝 날짜
        max_videos: 최대 동영상 개수
        collect_comments: 댓글 수집 여부
        order: 정렬 기준
    
    Returns:
        (동영상 데이터 리스트, 댓글 데이터 리스트) 튜플
    """
    try:
        all_videos_data = []
        all_comments_data = []
        seen_video_ids = set()
        
        print(f"\n{len(keywords)}개의 키워드로 YouTube 검색 중...")
        print(f"키워드: {', '.join(keywords[:5])}{'...' if len(keywords) > 5 else ''}")
        print(f"정렬 기준: {order}")
        
        # 각 키워드로 검색 수행
        for keyword_idx, keyword in enumerate(keywords, 1):
            if len(all_videos_data) >= max_videos:
                break
            
            print(f"\n키워드 {keyword_idx}/{len(keywords)}: '{keyword}' 검색 중...")
            
            # 키워드당 최대 수집 개수 계산
            remaining = max_videos - len(all_videos_data)
            per_keyword_limit = max(50, remaining // len(keywords) + 10)
            
            # 동영상 검색
            video_ids = search_videos(youtube, keyword, start_date, end_date, order, per_keyword_limit)
            
            if not video_ids:
                print(f"  '{keyword}' 검색 결과 없음")
                continue
            
            # 중복 제거
            new_video_ids = [vid for vid in video_ids if vid not in seen_video_ids]
            seen_video_ids.update(new_video_ids)
            
            if not new_video_ids:
                print(f"  '{keyword}': 중복 제외")
                continue
            
            # 동영상 상세 정보 수집
            videos_data = get_video_details(youtube, new_video_ids)
            
            # 날짜 필터링 및 검색 키워드 추가
            filtered_videos = []
            for video in videos_data:
                if video['upload_date']:
                    try:
                        video_date = datetime.strptime(video['upload_date'], '%Y-%m-%d')
                        if start_date <= video_date <= end_date:
                            video['search_keyword'] = keyword
                            filtered_videos.append(video)
                    except:
                        continue
                else:
                    video['search_keyword'] = keyword
                    filtered_videos.append(video)
            
            all_videos_data.extend(filtered_videos)
            
            print(f"  '{keyword}': {len(filtered_videos)}개 동영상 수집")
            
            # 댓글 수집 (옵션)
            if collect_comments:
                print(f"  댓글 수집 중...")
                for video in filtered_videos[:10]:  # 동영상당 최대 10개의 댓글만 수집 (API 할당량 고려)
                    if len(all_comments_data) >= max_videos * 10:  # 동영상당 평균 10개 댓글
                        break
                    
                    comments = get_video_comments(youtube, video['video_id'], max_comments=10)
                    all_comments_data.extend(comments)
                    
                    if comments:
                        print(f"    {video['title'][:30]}...: {len(comments)}개 댓글")
            
            if len(all_videos_data) >= max_videos:
                break
        
        # 최대 개수 제한
        all_videos_data = all_videos_data[:max_videos]
        
        print(f"\n총 {len(all_videos_data)}개의 동영상을 수집했습니다.")
        if collect_comments:
            print(f"총 {len(all_comments_data)}개의 댓글을 수집했습니다.")
        
        return all_videos_data, all_comments_data
    
    except Exception as e:
        raise Exception(f"데이터 수집 중 오류가 발생했습니다: {str(e)}")

def save_to_csv(videos_data, comments_data, keywords, start_date, end_date, collect_comments):
    """DataFrame을 CSV 및 Excel 파일로 저장"""
    try:
        if not videos_data:
            print("저장할 데이터가 없습니다.")
            return None, None, None, None, None
        
        # 출력 폴더 생성 (타임스탬프 포함)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_folder = f"crawl_output_{timestamp}"
        os.makedirs(output_folder, exist_ok=True)
        print(f"\n출력 폴더 생성: {output_folder}")
        
        # 파일명 생성
        first_keyword = keywords[0] if keywords else "search"
        sanitized_keyword = sanitize_filename(first_keyword)
        start_date_str = start_date.strftime("%Y%m%d")
        end_date_str = end_date.strftime("%Y%m%d")
        
        # 동영상 데이터 저장
        df_videos = pd.DataFrame(videos_data)
        
        # CSV 파일 저장
        videos_csv_filename = os.path.join(output_folder, f"youtube_videos_{sanitized_keyword}_{start_date_str}_{end_date_str}.csv")
        df_videos.to_csv(videos_csv_filename, index=False, encoding='utf-8-sig')
        print(f" 동영상 CSV 파일: {videos_csv_filename}")
        print(f"  총 {len(df_videos)}개의 동영상 데이터가 저장되었습니다.")
        
        # Excel 파일 저장
        videos_excel_filename = os.path.join(output_folder, f"youtube_videos_{sanitized_keyword}_{start_date_str}_{end_date_str}.xlsx")
        try:
            df_videos.to_excel(videos_excel_filename, index=False, engine='openpyxl')
            print(f"동영상 Excel 파일: {videos_excel_filename}")
        except ImportError:
            print("Excel 저장을 위해 openpyxl이 필요합니다: pip install openpyxl")
            videos_excel_filename = None
        except Exception as e:
            print(f"Excel 파일 저장 실패: {str(e)}")
            videos_excel_filename = None
        
        # 댓글 데이터 저장 (있는 경우)
        comments_csv_filename = None
        comments_excel_filename = None
        if collect_comments and comments_data:
            df_comments = pd.DataFrame(comments_data)
            
            # CSV 파일 저장
            comments_csv_filename = os.path.join(output_folder, f"youtube_comments_{sanitized_keyword}_{start_date_str}_{end_date_str}.csv")
            df_comments.to_csv(comments_csv_filename, index=False, encoding='utf-8-sig')
            print(f"✓ 댓글 CSV 파일: {comments_csv_filename}")
            print(f"  총 {len(df_comments)}개의 댓글 데이터가 저장되었습니다.")
            
            # Excel 파일 저장
            comments_excel_filename = os.path.join(output_folder, f"youtube_comments_{sanitized_keyword}_{start_date_str}_{end_date_str}.xlsx")
            try:
                df_comments.to_excel(comments_excel_filename, index=False, engine='openpyxl')
                print(f"✓ 댓글 Excel 파일: {comments_excel_filename}")
            except ImportError:
                print("Excel 저장을 위해 openpyxl이 필요합니다: pip install openpyxl")
            except Exception as e:
                print(f"댓글 Excel 파일 저장 실패: {str(e)}")
                comments_excel_filename = None
        
        return videos_csv_filename, videos_excel_filename, comments_csv_filename, comments_excel_filename, output_folder
    
    except Exception as e:
        raise Exception(f"파일 저장에 실패했습니다: {str(e)}")

def main():
    """메인 함수"""
    try:
        # 사용자 입력 받기
        media, all_keywords, korean_keywords, english_keywords, start_date, end_date, max_videos, collect_comments, order, use_oauth = get_user_input()
        
        # YouTube API 연결
        print("\nYouTube API 연결 중...")
        if use_oauth:
            print("OAuth 2.0 인증 방식을 사용합니다...")
        else:
            print("API 키 방식을 사용합니다...")
        youtube = connect_youtube(use_oauth)
        print("YouTube API 연결 성공!")
        
        # 크롤링 실행
        videos_data, comments_data = crawl_youtube(youtube, all_keywords, start_date, end_date, max_videos, collect_comments, order)
        
        if not videos_data:
            print("수집된 데이터가 없습니다.")
            return
        
        # CSV 및 Excel 파일로 저장
        videos_csv_filename, videos_excel_filename, comments_csv_filename, comments_excel_filename, output_folder = save_to_csv(
            videos_data, comments_data, all_keywords, start_date, end_date, collect_comments)
        
        # DataFrame 생성 및 출력 옵션 설정
        df_videos = pd.DataFrame(videos_data)
        
        # pandas 출력 옵션 설정 (더 자세한 표 형식으로)
        pd.set_option('display.max_columns', None)      # 모든 컬럼 표시
        pd.set_option('display.max_rows', None)         # 모든 행 표시
        pd.set_option('display.width', None)            # 출력 너비 제한 없음
        pd.set_option('display.max_colwidth', 50)       # 컬럼별 최대 너비 50자
        pd.set_option('display.expand_frame_repr', False)  # 프레임 표시 형식
        
        print("\n" + "=" * 100)
        print("수집된 동영상 데이터 전체")
        print("=" * 100)
        
        # 주요 컬럼만 선택하여 표 형식으로 출력 (너무 긴 컬럼은 제외)
        display_columns = ['video_id', 'title', 'channel_name', 'view_count', 'like_count', 
                          'comment_count', 'upload_date', 'search_keyword']
        available_columns = [col for col in display_columns if col in df_videos.columns]
        
        print(df_videos[available_columns].to_string(index=False))
        print(f"\n전체 동영상 개수: {len(df_videos)}")
        
        # 모든 데이터를 더 자세히 표시 (설명 포함, 최대 10개)
        print("\n" + "=" * 100)
        print("수집된 동영상 데이터 상세 (최대 10개)")
        print("=" * 100)
        
        # 설명(description) 컬럼은 길이 제한하여 표시
        df_display = df_videos.head(10).copy()
        if 'description' in df_display.columns:
            df_display['description'] = df_display['description'].apply(
                lambda x: x[:100] + '...' if isinstance(x, str) and len(x) > 100 else x)
        
        print(df_display.to_string(index=False))
        
        print(f"\n전체 동영상 개수: {len(df_videos)}")
        print(f"CSV 파일: {videos_csv_filename}")
        if videos_excel_filename:
            print(f"Excel 파일: {videos_excel_filename}")
        
        # 키워드별 통계
        if 'search_keyword' in df_videos.columns:
            print("\n" + "=" * 100)
            print("키워드별 수집된 동영상 개수")
            print("=" * 100)
            keyword_stats = df_videos['search_keyword'].value_counts()
            for keyword, count in keyword_stats.items():
                print(f"  '{keyword}': {count}개")
            
            # 키워드별 통계를 DataFrame으로 표시
            stats_df = pd.DataFrame({
                '키워드': keyword_stats.index,
                '개수': keyword_stats.values
            })
            print("\n키워드별 통계 (표 형식):")
            print(stats_df.to_string(index=False))
        
        # 댓글 통계
        if collect_comments and comments_data:
            df_comments = pd.DataFrame(comments_data)
            
            print("\n" + "=" * 100)
            print("수집된 댓글 데이터 전체")
            print("=" * 100)
            
            # 댓글 텍스트는 길이 제한
            df_comments_display = df_comments.copy()
            if 'text' in df_comments_display.columns:
                df_comments_display['text'] = df_comments_display['text'].apply(
                    lambda x: x[:80] + '...' if isinstance(x, str) and len(x) > 80 else x)
            
            print(df_comments_display.to_string(index=False))
            print(f"\n전체 댓글 개수: {len(df_comments)}")
            print(f"댓글 CSV 파일: {comments_csv_filename}")
            if comments_excel_filename:
                print(f"댓글 Excel 파일: {comments_excel_filename}")
        # 분석/시각화 실행 여부 묻기
        try:
            run_viz = input("\n수집된 데이터로 분석 및 시각화를 실행하시겠습니까? (y/n, 기본값: n): ").strip().lower()
        except Exception:
            run_viz = 'n'

        if run_viz in ('y', 'yes'):
            if videos_csv_filename:
                print("\n데이터 분석 및 시각화 실행 중...")
                try:
                    # 분석 보고서를 같은 폴더의 analysis_report 하위에 저장
                    analysis_output_dir = os.path.join(output_folder, 'analysis_report')
                    analyze_youtube_data(videos_csv_filename, output_dir=analysis_output_dir)
                except Exception as e:
                    print(f"분석 실행 중 오류: {str(e)}")
            else:
                print("CSV 파일이 없어 분석을 실행할 수 없습니다.")
    except ValueError as e:
        print(f"\n입력 오류: {str(e)}")
    except ConnectionError as e:
        print(f"\n연결 오류: {str(e)}")
    except Exception as e:
        print(f"\n오류 발생: {str(e)}")

if __name__ == "__main__":
    main()


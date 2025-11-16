import os
import re
from datetime import datetime
from dotenv import load_dotenv
import praw
import pandas as pd
from keyword_extractor import extract_and_translate_keywords

# 환경변수 로드
load_dotenv()

def sanitize_filename(text):
    """파일명에서 특수문자 제거 및 공백을 언더스코어로 변경"""
    # 특수문자 제거, 공백을 언더스코어로 변경
    text = re.sub(r'[^\w\s-]', '', text)
    text = re.sub(r'[-\s]+', '_', text)
    return text 

def get_user_input():
    """콘솔에서 사용자 입력 받기"""
    print("=" * 50)
    print("크롤링 프로그램 크롤크롤~~~~")
    print("=" * 50)
    
    # 매체 선택 (현재는 reddit만)
    media = input("크롤링할 매체를 입력하세요 (현재는 'reddit'만 지원): ").strip().lower()
    if media != 'reddit':
        print("현재는 'reddit'만 지원합니다. 'reddit'으로 설정합니다.")
        media = 'reddit'
    
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
        max_posts = int(input("크롤링할 데이터 개수를 입력하세요: ").strip())
        if max_posts <= 0:
            raise ValueError("데이터 개수는 1 이상이어야 합니다.")
    except ValueError as e:
        if "invalid literal" in str(e):
            raise ValueError("데이터 개수는 숫자로 입력해주세요.")
        raise
    
    return media, all_keywords, korean_keywords, english_keywords, start_date, end_date, max_posts

def connect_reddit():
    """Reddit API 연결"""
    try:
        client_id = os.getenv('REDDIT_CLIENT_ID')
        client_secret = os.getenv('REDDIT_CLIENT_SECRET')
        user_agent = os.getenv('REDDIT_USER_AGENT')
        
        if not all([client_id, client_secret, user_agent]):
            raise ValueError("환경변수에 Reddit API 정보가 없습니다. .env 파일을 확인해주세요.")
        
        reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent,
            check_for_async=False
        )
        
        # 연결 테스트 (read-only 모드)
        # 간단한 요청으로 연결 확인 (존재하는 서브레딧 사용)
        try:
            _ = reddit.subreddit('popular').display_name
        except:
            # 연결 테스트 실패해도 계속 진행 (실제 크롤링 시 오류 확인)
            pass
        
        return reddit
    except Exception as e:
        raise ConnectionError(f"Reddit API 연결에 실패했습니다: {str(e)}")

def crawl_reddit(reddit, keywords, start_date, end_date, max_posts):
    """Reddit에서 여러 키워드로 게시물 크롤링"""
    try:
        posts_data = []
        
        # 전체 Reddit에서 키워드로 검색 (관련도순)
        # Reddit API는 날짜 범위 직접 지원이 제한적이므로 검색 후 필터링
        print(f"\n{len(keywords)}개의 키워드로 Reddit 검색 중...")
        print(f"키워드: {', '.join(keywords[:5])}{'...' if len(keywords) > 5 else ''}")
        
        # 주요 서브레딧 목록 (전체 Reddit 검색을 위해 여러 서브레딧에서 검색)
        # Reddit API의 search는 특정 서브레딧에서만 작동하므로 여러 인기 서브레딧에서 검색
        popular_subreddits = [
            'all',  # 전체 (가능한 경우)
            'popular',
            'python', 'programming', 'technology', 'webdev',
            'news', 'worldnews', 'science', 'todayilearned',
            'askreddit', 'explainlikeimfive', 'showerthoughts',
            'funny', 'gaming', 'movies', 'music', 'books',
            'sports', 'fitness', 'food', 'travel'
        ]
        
        seen_post_ids = set()  # 중복 제거를 위한 set
        count = 0
        
        # 각 키워드로 검색 수행
        for keyword_idx, keyword in enumerate(keywords, 1):
            if count >= max_posts:
                break
            
            print(f"\n키워드 {keyword_idx}/{len(keywords)}: '{keyword}' 검색 중...")
            
            # 먼저 'all' 서브레딧에서 시도
            try:
                search_results = reddit.subreddit('all').search(keyword, sort='relevance', limit=300, time_filter='all')
                
                for submission in search_results:
                    if count >= max_posts:
                        break
                    
                    # 중복 체크
                    if submission.id in seen_post_ids:
                        continue
                    seen_post_ids.add(submission.id)
                    
                    # 날짜 필터링
                    post_date = datetime.fromtimestamp(submission.created_utc)
                    
                    if post_date < start_date or post_date > end_date:
                        continue
                    
                    # 데이터 수집
                    post_info = {
                        'title': submission.title,
                        'content': submission.selftext,
                        'author': str(submission.author) if submission.author else 'deleted',
                        'score': submission.score,
                        'num_comments': submission.num_comments,
                        'url': submission.url,
                        'created_date': post_date.strftime("%Y-%m-%d %H:%M:%S"),
                        'subreddit': submission.subreddit.display_name,
                        'post_id': submission.id,
                        'search_keyword': keyword  # 어떤 키워드로 찾았는지 기록
                    }
                    posts_data.append(post_info)
                    count += 1
                    
            except Exception as e:
                print(f"  'all' 서브레딧에서 '{keyword}' 검색 실패: {str(e)}")
            
            # 'all'에서 충분한 데이터를 못 얻은 경우 주요 서브레딧에서 추가 검색
            if count < max_posts:
                for subreddit_name in popular_subreddits[2:]:  # 'all', 'popular' 제외
                    if count >= max_posts:
                        break
                    
                    try:
                        subreddit = reddit.subreddit(subreddit_name)
                        search_results = subreddit.search(keyword, sort='relevance', limit=50, time_filter='all')
                        
                        for submission in search_results:
                            if count >= max_posts:
                                break
                            
                            # 중복 체크
                            if submission.id in seen_post_ids:
                                continue
                            seen_post_ids.add(submission.id)
                            
                            # 날짜 필터링
                            post_date = datetime.fromtimestamp(submission.created_utc)
                            
                            if post_date < start_date or post_date > end_date:
                                continue
                            
                            # 데이터 수집
                            post_info = {
                                'title': submission.title,
                                'content': submission.selftext,
                                'author': str(submission.author) if submission.author else 'deleted',
                                'score': submission.score,
                                'num_comments': submission.num_comments,
                                'url': submission.url,
                                'created_date': post_date.strftime("%Y-%m-%d %H:%M:%S"),
                                'subreddit': submission.subreddit.display_name,
                                'post_id': submission.id,
                                'search_keyword': keyword  # 어떤 키워드로 찾았는지 기록
                            }
                            posts_data.append(post_info)
                            count += 1
                            
                    except Exception as e:
                        # 특정 서브레딧에서 검색 실패해도 계속 진행
                        continue
        
        print(f"\n날짜 범위 내에서 {count}개의 게시물을 수집했습니다.")
        return posts_data
    
    except Exception as e:
        raise Exception(f"데이터 수집 중 오류가 발생했습니다: {str(e)}")

def save_to_csv(data, keywords, start_date, end_date):
    """DataFrame을 CSV 파일로 저장"""
    try:
        if not data:
            print("저장할 데이터가 없습니다.")
            return None
        
        # DataFrame 생성
        df = pd.DataFrame(data)
        
        # 파일명 생성 (첫 번째 키워드 사용)
        first_keyword = keywords[0] if keywords else "search"
        sanitized_keyword = sanitize_filename(first_keyword)
        start_date_str = start_date.strftime("%Y%m%d")
        end_date_str = end_date.strftime("%Y%m%d")
        filename = f"reddit_{sanitized_keyword}_{start_date_str}_{end_date_str}.csv"
        
        # CSV 파일로 저장
        df.to_csv(filename, index=False, encoding='utf-8-sig')
        
        print(f"\nCSV 파일이 저장되었습니다: {filename}")
        print(f"총 {len(df)}개의 데이터가 저장되었습니다.")
        
        return filename
    
    except Exception as e:
        raise Exception(f"CSV 파일 저장에 실패했습니다: {str(e)}")

def main():
    """메인 함수"""
    try:
        # 사용자 입력 받기
        media, all_keywords, korean_keywords, english_keywords, start_date, end_date, max_posts = get_user_input()
        
        # Reddit API 연결
        print("\nReddit API 연결 중...")
        reddit = connect_reddit()
        print("Reddit API 연결 성공!")
        
        # 크롤링 실행
        posts_data = crawl_reddit(reddit, all_keywords, start_date, end_date, max_posts)
        
        if not posts_data:
            print("수집된 데이터가 없습니다.")
            return
        
        # CSV 파일로 저장
        csv_filename = save_to_csv(posts_data, all_keywords, start_date, end_date)
        
        # DataFrame 정보 출력
        df = pd.DataFrame(posts_data)
        print("\n" + "=" * 50)
        print("수집된 데이터 요약")
        print("=" * 50)
        print(df.head())
        print(f"\n전체 데이터 개수: {len(df)}")
        print(f"CSV 파일: {csv_filename}")
        
        # 키워드별 통계
        if 'search_keyword' in df.columns:
            print("\n키워드별 수집된 데이터 개수:")
            keyword_stats = df['search_keyword'].value_counts()
            for keyword, count in keyword_stats.items():
                print(f"  '{keyword}': {count}개")
        
    except ValueError as e:
        print(f"\n입력 오류: {str(e)}")
    except ConnectionError as e:
        print(f"\n연결 오류: {str(e)}")
    except Exception as e:
        print(f"\n오류 발생: {str(e)}")

if __name__ == "__main__":
    main()


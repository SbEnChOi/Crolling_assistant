"""
snscrape를 사용한 Reddit 크롤링 프로그램
API 키 없이 Reddit 데이터를 수집합니다.
"""

import os
import re
import json
import subprocess
from datetime import datetime
import pandas as pd
from keyword_extractor import extract_and_translate_keywords

def sanitize_filename(text):
    """파일명에서 특수문자 제거 및 공백을 언더스코어로 변경"""
    text = re.sub(r'[^\w\s-]', '', text)
    text = re.sub(r'[-\s]+', '_', text)
    return text

def get_user_input():
    """콘솔에서 사용자 입력 받기"""
    print("=" * 50)
    print("크롤링 프로그램 (snscrape 사용)")
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


def crawl_reddit_snscrape(keywords, start_date, end_date, max_posts):
    """
    snscrape를 사용하여 Reddit에서 게시물 크롤링
    
    Args:
        keywords: 검색할 키워드 리스트
        start_date: 시작 날짜
        end_date: 끝 날짜
        max_posts: 최대 수집 개수
    
    Returns:
        수집된 게시물 데이터 리스트
    """
    try:
        import snscrape.modules.reddit as snreddit
    except ImportError:
        raise ImportError("snscrape가 설치되지 않았습니다. 'pip install snscrape'를 실행하세요.")
    
    posts_data = []
    seen_post_ids = set()
    count = 0
    
    print(f"\n{len(keywords)}개의 키워드로 Reddit 검색 중...")
    print(f"키워드: {', '.join(keywords[:5])}{'...' if len(keywords) > 5 else ''}")
    
    # 날짜 형식 변환 (YYYY-MM-DD)
    start_date_str = start_date.strftime("%Y-%m-%d")
    end_date_str = end_date.strftime("%Y-%m-%d")
    
    # 각 키워드로 검색 수행
    for keyword_idx, keyword in enumerate(keywords, 1):
        if count >= max_posts:
            break
        
        print(f"\n키워드 {keyword_idx}/{len(keywords)}: '{keyword}' 검색 중...")
        
        try:
            # Reddit 검색 쿼리 생성
            # snscrape의 Reddit 검색 형식: reddit-search:{keyword} since:{start_date} until:{end_date}
            query = f"{keyword} subreddit:all since:{start_date_str} until:{end_date_str}"
            
            # snscrape로 Reddit 검색
            scraper = snreddit.RedditSearchScraper(query)
            
            keyword_count = 0
            for post in scraper.get_items():
                if count >= max_posts:
                    break
                
                # 중복 체크
                if post.id in seen_post_ids:
                    continue
                seen_post_ids.add(post.id)
                
                # 날짜 필터링 (추가 확인)
                post_date = post.date.replace(tzinfo=None) if post.date else None
                if post_date:
                    if post_date < start_date or post_date > end_date:
                        continue
                
                # 데이터 수집
                post_info = {
                    'title': post.title if hasattr(post, 'title') else '',
                    'content': post.content if hasattr(post, 'content') else post.selftext if hasattr(post, 'selftext') else '',
                    'author': post.author if hasattr(post, 'author') else 'deleted',
                    'score': post.score if hasattr(post, 'score') else 0,
                    'num_comments': post.commentCount if hasattr(post, 'commentCount') else post.num_comments if hasattr(post, 'num_comments') else 0,
                    'url': post.url if hasattr(post, 'url') else f"https://www.reddit.com{post.permalink}" if hasattr(post, 'permalink') else '',
                    'created_date': post_date.strftime("%Y-%m-%d %H:%M:%S") if post_date else '',
                    'subreddit': post.subreddit if hasattr(post, 'subreddit') else 'unknown',
                    'post_id': post.id if hasattr(post, 'id') else '',
                    'search_keyword': keyword
                }
                posts_data.append(post_info)
                count += 1
                keyword_count += 1
                
                if keyword_count >= max_posts // len(keywords) + 10:  # 키워드당 최대 수집 개수
                    break
            
            print(f"  '{keyword}': {keyword_count}개 수집")
            
        except Exception as e:
            print(f"  '{keyword}' 검색 중 오류 발생: {str(e)}")
            # 다른 방법으로 시도 (서브레딧별 검색)
            try:
                popular_subreddits = ['python', 'programming', 'technology', 'news', 'worldnews', 'science']
                for subreddit_name in popular_subreddits[:3]:  # 최대 3개 서브레딧만 시도
                    if count >= max_posts:
                        break
                    
                    try:
                        query = f"{keyword} subreddit:{subreddit_name} since:{start_date_str} until:{end_date_str}"
                        scraper = snreddit.RedditSearchScraper(query)
                        
                        for post in scraper.get_items():
                            if count >= max_posts:
                                break
                            
                            if post.id in seen_post_ids:
                                continue
                            seen_post_ids.add(post.id)
                            
                            post_date = post.date.replace(tzinfo=None) if post.date else None
                            if post_date and (post_date < start_date or post_date > end_date):
                                continue
                            
                            post_info = {
                                'title': post.title if hasattr(post, 'title') else '',
                                'content': post.content if hasattr(post, 'content') else post.selftext if hasattr(post, 'selftext') else '',
                                'author': post.author if hasattr(post, 'author') else 'deleted',
                                'score': post.score if hasattr(post, 'score') else 0,
                                'num_comments': post.commentCount if hasattr(post, 'commentCount') else post.num_comments if hasattr(post, 'num_comments') else 0,
                                'url': post.url if hasattr(post, 'url') else f"https://www.reddit.com{post.permalink}" if hasattr(post, 'permalink') else '',
                                'created_date': post_date.strftime("%Y-%m-%d %H:%M:%S") if post_date else '',
                                'subreddit': post.subreddit if hasattr(post, 'subreddit') else 'unknown',
                                'post_id': post.id if hasattr(post, 'id') else '',
                                'search_keyword': keyword
                            }
                            posts_data.append(post_info)
                            count += 1
                    except:
                        continue
            except:
                continue
    
    print(f"\n날짜 범위 내에서 {count}개의 게시물을 수집했습니다.")
    return posts_data


def crawl_reddit_alternative(keywords, start_date, end_date, max_posts):
    """
    snscrape 명령줄 도구를 사용하여 Reddit 크롤링 (대안 방법)
    
    Args:
        keywords: 검색할 키워드 리스트
        start_date: 시작 날짜
        end_date: 끝 날짜
        max_posts: 최대 수집 개수
    
    Returns:
        수집된 게시물 데이터 리스트
    """
    posts_data = []
    seen_post_ids = set()
    count = 0
    
    print(f"\n{len(keywords)}개의 키워드로 Reddit 검색 중 (명령줄 방식)...")
    print(f"키워드: {', '.join(keywords[:5])}{'...' if len(keywords) > 5 else ''}")
    
    start_date_str = start_date.strftime("%Y-%m-%d")
    end_date_str = end_date.strftime("%Y-%m-%d")
    
    for keyword_idx, keyword in enumerate(keywords, 1):
        if count >= max_posts:
            break
        
        print(f"\n키워드 {keyword_idx}/{len(keywords)}: '{keyword}' 검색 중...")
        
        try:
            # snscrape 명령줄 도구 사용
            query = f"reddit-submission {keyword} since:{start_date_str} until:{end_date_str}"
            
            # subprocess로 snscrape 실행
            process = subprocess.Popen(
                ['snscrape', '--jsonl', query],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding='utf-8'
            )
            
            keyword_count = 0
            for line in process.stdout:
                if count >= max_posts:
                    break
                
                try:
                    post_data = json.loads(line.strip())
                    post_id = post_data.get('id', '')
                    
                    if post_id in seen_post_ids:
                        continue
                    seen_post_ids.add(post_id)
                    
                    # 날짜 파싱
                    created_utc = post_data.get('created_utc', 0)
                    if created_utc:
                        post_date = datetime.fromtimestamp(created_utc)
                        if post_date < start_date or post_date > end_date:
                            continue
                    
                    # 데이터 수집
                    post_info = {
                        'title': post_data.get('title', ''),
                        'content': post_data.get('selftext', ''),
                        'author': post_data.get('author', {}).get('name', 'deleted') if isinstance(post_data.get('author'), dict) else post_data.get('author', 'deleted'),
                        'score': post_data.get('score', 0),
                        'num_comments': post_data.get('num_comments', 0),
                        'url': post_data.get('url', ''),
                        'created_date': post_date.strftime("%Y-%m-%d %H:%M:%S") if created_utc else '',
                        'subreddit': post_data.get('subreddit', {}).get('name', 'unknown') if isinstance(post_data.get('subreddit'), dict) else post_data.get('subreddit', 'unknown'),
                        'post_id': post_id,
                        'search_keyword': keyword
                    }
                    posts_data.append(post_info)
                    count += 1
                    keyword_count += 1
                    
                except (json.JSONDecodeError, KeyError, ValueError) as e:
                    continue
            
            process.wait()
            print(f"  '{keyword}': {keyword_count}개 수집")
            
        except FileNotFoundError:
            raise FileNotFoundError("snscrape가 설치되지 않았거나 PATH에 없습니다. 'pip install snscrape'를 실행하세요.")
        except Exception as e:
            print(f"  '{keyword}' 검색 중 오류 발생: {str(e)}")
            continue
    
    print(f"\n날짜 범위 내에서 {count}개의 게시물을 수집했습니다.")
    return posts_data


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
        filename = f"reddit_snscrape_{sanitized_keyword}_{start_date_str}_{end_date_str}.csv"
        
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
        
        # snscrape를 사용한 크롤링 시도
        print("\nsnscrape로 Reddit 크롤링 시작...")
        
        try:
            # 먼저 Python 라이브러리 방식 시도
            posts_data = crawl_reddit_snscrape(all_keywords, start_date, end_date, max_posts)
        except (ImportError, AttributeError) as e:
            print(f"Python 라이브러리 방식 실패: {str(e)}")
            print("명령줄 방식으로 시도합니다...")
            try:
                # 명령줄 도구 방식 시도
                posts_data = crawl_reddit_alternative(all_keywords, start_date, end_date, max_posts)
            except Exception as e2:
                raise Exception(f"snscrape 크롤링 실패: {str(e2)}\n'snscrape'가 설치되어 있는지 확인하세요: pip install snscrape")
        
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
    except Exception as e:
        print(f"\n오류 발생: {str(e)}")


if __name__ == "__main__":
    main()


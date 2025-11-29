"""
YouTube 크롤링 및 데이터 분석 통합 프로그램
이 파일은 youtube 크롤링, 키워드 추출, 데이터 분석 기능을 하나로 통합한 파일입니다.
"""

import os
import re
import sys
import pickle
import warnings
from datetime import datetime
from typing import List, Tuple, Set

# Third-party imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from matplotlib import font_manager, rc
from dotenv import load_dotenv
from konlpy.tag import Okt
from deep_translator import GoogleTranslator
import nltk
from nltk.corpus import stopwords
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# ---------------------------------------------------------
# Keyword Extractor Code
# ---------------------------------------------------------

# NLTK 데이터 다운로드 (최초 1회만)
try:
    nltk.data.find('tokenizers/punkt')
except (LookupError, Exception):
    try:
        nltk.download('punkt', quiet=True)
    except Exception:
        pass

# 영어 불용어 로드
ENGLISH_STOPWORDS = set()
try:
    nltk.data.find('corpora/stopwords')
except (LookupError, Exception):
    try:
        nltk.download('stopwords', quiet=True)
    except Exception:
        pass

try:
    ENGLISH_STOPWORDS = set(stopwords.words('english'))
except Exception:
    # NLTK 로드 실패 시 기본 불용어 사용
    ENGLISH_STOPWORDS = {
        'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
        'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
        'to', 'was', 'will', 'with', 'the', 'this', 'but', 'they', 'have',
        'had', 'what', 'said', 'each', 'which', 'their', 'time', 'if',
        'up', 'out', 'many', 'then', 'them', 'these', 'so', 'some', 'her',
        'would', 'make', 'like', 'into', 'him', 'has', 'two', 'more', 'go',
        'no', 'way', 'could', 'my', 'than', 'first', 'been', 'call', 'who',
        'oil', 'sit', 'now', 'find', 'down', 'day', 'did', 'get', 'come',
        'made', 'may', 'part'
    }

# 한국어 불용어 및 조사 목록
KOREAN_STOPWORDS = {
    '이', '가', '을', '를', '에', '에서', '으로', '로', '의', '과', '와', '도', '만', '부터', '까지',
    '는', '은', '에게', '께', '한테', '더', '또', '그', '그것', '이것', '저것', '그런', '이런', '저런',
    '그리고', '그러나', '하지만', '또한', '그래서', '따라서', '그런데', '그럼', '그렇다면',
    '있다', '없다', '하다', '되다', '이다', '아니다', '것', '수', '때', '곳', '거', '게',
    '등', '등등', '및', '또는', '혹은', '만약', '만일',
    # 조사 추가
    '에게서', '한테서', '께서', '처럼', '만큼', '처럼', '보다', '같이', '커녕', '마저', '조차',
    '든지', '이나', '든가', '라도', '이라도', '이라도', '부터는', '부터도'
}

# 조사 패턴 (정규표현식)
JOSA_PATTERNS = [
    r'이$', r'가$', r'을$', r'를$', r'에$', r'에서$', r'으로$', r'로$', r'의$', r'과$', r'와$',
    r'는$', r'은$', r'도$', r'만$', r'부터$', r'까지$', r'에게$', r'께$', r'한테$',
    r'에서$', r'으로부터$', r'처럼$', r'만큼$', r'보다$', r'같이$', r'마저$', r'조차$',
    r'든지$', r'이나$', r'든가$', r'라도$', r'이라도$', r'부터는$', r'부터도$'
]

class KeywordExtractor:
    """키워드 추출 클래스"""
    
    def __init__(self):
        try:
            self.okt = Okt()
        except Exception as e:
            print(f"KoNLPy 초기화 실패: {str(e)}")
            self.okt = None
        
        try:
            self.translator = GoogleTranslator(source='ko', target='en')
        except Exception as e:
            print(f"번역기 초기화 실패: {str(e)}")
            self.translator = None
    
    def extract_keywords(self, sentence: str, min_length: int = 2) -> List[str]:
        """
        문장에서 키워드를 추출합니다.
        
        Args:
            sentence: 입력 문장
            min_length: 최소 키워드 길이 (기본값: 2)
        
        Returns:
            추출된 키워드 리스트
        """
        if not sentence or not sentence.strip():
            return []
        
        sentence = sentence.strip()
        
        # 매우 짧은 문장 (1-2단어)인 경우 전체를 키워드로 반환
        words = sentence.split()
        if len(words) <= 2:
            # 공백 제거 후 반환
            return [sentence.replace(' ', '')] if sentence.replace(' ', '') else []
        
        # 형태소 분석 (명사, 동사, 형용사 추출)
        if self.okt is None:
            # KoNLPy가 없는 경우 간단한 토큰화
            words = sentence.split()
            keywords = [w for w in words if w not in KOREAN_STOPWORDS and len(w) >= min_length and not w.isdigit()]
            return keywords if keywords else [sentence]
        
        try:
            # 형태소 분석 (명사, 동사, 형용사만 추출)
            pos_tags = self.okt.pos(sentence, norm=True, stem=True)
            keywords = []
            
            for word, pos in pos_tags:
                # 조사(J)는 제외
                if pos.startswith('J'):
                    continue
                
                # 명사, 동사, 형용사만 추출
                if pos.startswith('N') or pos.startswith('V') or pos.startswith('A'):
                    # 불용어 제거
                    if word not in KOREAN_STOPWORDS and len(word) >= min_length:
                        # 숫자만 있는 경우 제외
                        if not word.isdigit():
                            # 조사가 붙어있는지 확인하고 제거
                            cleaned_word = self._remove_josa(word)
                            if cleaned_word and len(cleaned_word) >= min_length:
                                keywords.append(cleaned_word)
            
            # 중복 제거 및 정렬
            keywords = sorted(list(set(keywords)), key=len, reverse=True)
            
            # 키워드가 없으면 원문 반환
            if not keywords:
                return [sentence]
            
            return keywords
            
        except Exception as e:
            # 형태소 분석 실패 시 원문 반환
            print(f"키워드 추출 중 오류 발생: {str(e)}")
            return [sentence]
    
    def _remove_josa(self, word: str) -> str:
        """
        단어에서 조사를 제거.
        Args:
            word: 조사가 붙을 수 있는 단어
        
        Returns:
            조사가 제거된 단어
        """
        if not word:
            return word
        
        # 먼저 조사 목록에 있는지 확인
        if word in KOREAN_STOPWORDS:
            return ''
        
        # 조사 패턴 제거 (맨 끝에 붙은 조사)
        cleaned = word
        for pattern in JOSA_PATTERNS:
            cleaned = re.sub(pattern, '', cleaned)
            # 패턴 제거 후 변경이 있었으면 다시 확인
            if cleaned != word:
                break
        
        # 조사 패턴 제거 후에도 조사가 남아있는지 확인
        # 예: "파이썬을을" 같은 경우
        max_iterations = 3
        iteration = 0
        while iteration < max_iterations:
            prev_cleaned = cleaned
            for pattern in JOSA_PATTERNS:
                cleaned = re.sub(pattern, '', cleaned)
            if cleaned == prev_cleaned:
                break
            iteration += 1
        
        # 빈 문자열이 되거나 너무 짧아지면 원본 반환 (하지만 조사만 있는 경우는 제외)
        if not cleaned or len(cleaned) < 2:
            # 원본이 조사 목록에 있으면 빈 문자열 반환
            if word in KOREAN_STOPWORDS:
                return ''
            # 원본이 조사로 끝나지만 조사가 아닌 경우 (예: "것"은 명사이지만 조사 목록에 있음)
            # 형태소 분석에서 이미 걸러졌을 것이므로 원본 반환
            return word if len(word) >= 2 else ''
        
        # 조사가 제거된 단어가 조사 목록에 있는지 확인
        if cleaned in KOREAN_STOPWORDS:
            return ''
        
        return cleaned
    
    def translate_keywords(self, keywords: List[str]) -> List[str]:
        """
        한국어 키워드를 영어로 번역합니다.
        
        Args:
            keywords: 한국어 키워드 리스트
        
        Returns:
            영어로 번역된 키워드 리스트
        """
        if self.translator is None:
            # 번역기가 없는 경우 영어인 키워드만 반환
            return [kw.lower() for kw in keywords if self._is_english(kw)]
        
        translated_keywords = []
        
        for keyword in keywords:
            # 이미 영어인 경우
            if self._is_english(keyword):
                translated_keywords.append(keyword.lower())
                continue
            
            try:
                # 영어 번역
                translated = self.translator.translate(keyword)
                if translated and translated != keyword:
                    translated_keywords.append(translated.lower())
            except Exception as e:
                # 번역 실패 시 무시 (한국어 키워드만 사용)
                pass
        
        return translated_keywords
    
    def _is_english(self, text: str) -> bool:
        """텍스트가 영어인지 확인"""
        try:
            text.encode('ascii')
            return True
        except UnicodeEncodeError:
            return False
    
    def filter_english_keywords(self, keywords: List[str]) -> List[str]:
        """영어 키워드에서 불용어 제거"""
        filtered = []
        for keyword in keywords:
            # 단어 단위로 분리
            words = keyword.split()
            filtered_words = [w for w in words if w.lower() not in ENGLISH_STOPWORDS and len(w) >= 2]
            if filtered_words:
                filtered.append(' '.join(filtered_words))
        return filtered if filtered else keywords


def select_keywords(keywords: List[str]) -> List[str]:
    """
    사용자에게 키워드를 선택하게 합니다.
    
    Args:
        keywords: 추출된 키워드 리스트
    
    Returns:
        사용자가 선택한 키워드 리스트
    """
    if not keywords:
        return []
    
    print("\n추출된 키워드 목록:")
    print("-" * 50)
    for keyword in keywords:
        print(f"  - {keyword}")
    print("-" * 50)
    
    while True:
        try:
            selection = input("\n사용할 키워드를 입력하세요 (키워드를 콤마로 구분, 예: 파이썬,프로그래밍,학습 또는 'all' 전체 선택): ").strip()
            
            if selection.lower() == 'all':
                return keywords
            
            if not selection:
                print("키워드를 입력해주세요.")
                continue
            
            # 콤마로 구분된 키워드 파싱
            input_keywords = [kw.strip() for kw in selection.split(',') if kw.strip()]
            
            if not input_keywords:
                print("유효한 키워드가 없습니다. 다시 입력해주세요.")
                continue
            
            # 입력된 키워드가 추출된 키워드 목록에 있는지 확인
            selected_keywords = []
            not_found_keywords = []
            
            for input_kw in input_keywords:
                # 정확히 일치하는 키워드 찾기
                found = False
                for kw in keywords:
                    if kw.lower() == input_kw.lower() or kw == input_kw:
                        selected_keywords.append(kw)  # 원본 키워드 사용 (대소문자 유지)
                        found = True
                        break
                
                if not found:
                    not_found_keywords.append(input_kw)
            
            # 찾지 못한 키워드가 있으면 경고
            if not_found_keywords:
                print(f"경고: 다음 키워드를 찾을 수 없습니다: {', '.join(not_found_keywords)}")
                print("추출된 키워드 목록에서 정확히 입력해주세요.")
            
            # 선택된 키워드가 있으면 반환
            if selected_keywords:
                # 중복 제거 (순서 유지)
                seen = set()
                unique_selected = []
                for kw in selected_keywords:
                    if kw not in seen:
                        seen.add(kw)
                        unique_selected.append(kw)
                
                print(f"\n선택된 키워드: {', '.join(unique_selected)}")
                return unique_selected
            else:
                print("선택된 키워드가 없습니다. 다시 입력해주세요.")
                
        except KeyboardInterrupt:
            print("\n취소되었습니다.")
            return []
        except Exception as e:
            print(f"오류 발생: {str(e)}")
            return keywords


def extract_and_translate_keywords(sentence: str) -> Tuple[List[str], List[str]]:
    """
    문장에서 키워드를 추출하고 영어로 번역합니다.
    
    Args:
        sentence: 입력 문장
    
    Returns:
        (한국어 키워드 리스트, 영어 키워드 리스트) 튜플
    """
    extractor = KeywordExtractor()
    
    # 키워드 추출
    korean_keywords = extractor.extract_keywords(sentence)
    
    # 사용자가 키워드 선택
    selected_korean_keywords = select_keywords(korean_keywords)
    
    if not selected_korean_keywords:
        # 선택된 키워드가 없으면 원문 사용
        selected_korean_keywords = [sentence]
    
    # 영어 번역
    english_keywords = extractor.translate_keywords(selected_korean_keywords)
    
    # 영어 키워드에서 불용어 제거
    english_keywords = extractor.filter_english_keywords(english_keywords)
    
    return selected_korean_keywords, english_keywords

# ---------------------------------------------------------
# Data Analysis Code
# ---------------------------------------------------------

warnings.filterwarnings('ignore')

# 한글 폰트 설정
sns.set_style("whitegrid")
sns.set_palette("husl")
plt.rcParams['font.family'] = 'Malgun Gothic'


class YouTubeAnalyzer:
    """YouTube 수집 데이터 분석 클래스"""
    
    def __init__(self, csv_file):
        """
        분석기 초기화
        
        Args:
            csv_file: YouTube 동영상 데이터 CSV 파일 경로
        """
        self.df = pd.read_csv(csv_file, encoding='utf-8-sig')
        self.df['upload_date'] = pd.to_datetime(self.df['upload_date'], errors='coerce')
        self.df['view_count'] = pd.to_numeric(self.df['view_count'], errors='coerce')
        self.df['like_count'] = pd.to_numeric(self.df['like_count'], errors='coerce')
        self.df['comment_count'] = pd.to_numeric(self.df['comment_count'], errors='coerce')
        
        print(f"✓ {len(self.df)}개의 동영상 데이터 로드 완료")
        print(f"  - 데이터 기간: {self.df['upload_date'].min()} ~ {self.df['upload_date'].max()}")
    
    def get_summary_statistics(self):
        """기본 통계 정보 반환"""
        stats = {
            '총 동영상 수': len(self.df),
            '총 조회수': f"{self.df['view_count'].sum():,.0f}",
            '평균 조회수': f"{self.df['view_count'].mean():,.0f}",
            '중앙값 조회수': f"{self.df['view_count'].median():,.0f}",
            '최고 조회수': f"{self.df['view_count'].max():,.0f}",
            '총 좋아요': f"{self.df['like_count'].sum():,.0f}",
            '평균 좋아요': f"{self.df['like_count'].mean():,.0f}",
            '총 댓글': f"{self.df['comment_count'].sum():,.0f}",
            '평균 댓글': f"{self.df['comment_count'].mean():,.0f}",
            '채널 수': self.df['channel_name'].nunique(),
        }
        return stats
    
    def print_summary_statistics(self):
        """기본 통계 정보 출력"""
        stats = self.get_summary_statistics()
        print("\n" + "="*50)
        print("기본 통계")
        print("="*50)
        for key, value in stats.items():
            print(f"{key:.<30} {value}")
    
    def plot_top_videos(self, n=10, figsize=(14, 8)):
        """상위 N개 인기 동영상 (조회수 기준)"""
        top_videos = self.df.nlargest(n, 'view_count')[['title', 'view_count', 'like_count', 'comment_count']]
        top_videos['title'] = top_videos['title'].str[:50] + '...'
        
        fig, ax = plt.subplots(figsize=figsize)
        
        y_pos = np.arange(len(top_videos))
        ax.barh(y_pos, top_videos['view_count'].values, color='#1f77b4', label='조회수')
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_videos['title'].values, fontsize=10)
        ax.set_xlabel('조회수', fontsize=12, fontweight='bold')
        ax.set_title(f'상위 {n}개 인기 동영상 (조회수 기준)', fontsize=14, fontweight='bold', pad=20)
        ax.invert_yaxis()
        
        # 값 표시
        for i, v in enumerate(top_videos['view_count'].values):
            ax.text(v, i, f' {int(v):,}', va='center', fontsize=9)
        
        plt.tight_layout()
        return fig
    
    def plot_top_channels(self, n=10, figsize=(12, 6)):
        """채널별 동영상 개수 TOP-N"""
        top_channels = self.df['channel_name'].value_counts().head(n)
        
        fig, ax = plt.subplots(figsize=figsize)
        colors = sns.color_palette("husl", n)
        bars = ax.bar(range(len(top_channels)), top_channels.values, color=colors)
        
        ax.set_xticks(range(len(top_channels)))
        ax.set_xticklabels(top_channels.index, rotation=45, ha='right', fontsize=10)
        ax.set_ylabel('동영상 개수', fontsize=12, fontweight='bold')
        ax.set_title(f'상위 {n}개 채널 (동영상 개수)', fontsize=14, fontweight='bold', pad=20)
        
        # 값 표시
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        return fig
    
    def plot_channel_statistics(self, n=10, figsize=(14, 8)):
        """채널별 통계 (조회수, 좋아요, 댓글)"""
        channel_stats = self.df.groupby('channel_name').agg({
            'view_count': 'sum',
            'like_count': 'sum',
            'comment_count': 'sum'
        }).nlargest(n, 'view_count')
        
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # 조회수
        channel_stats['view_count'].plot(kind='barh', ax=axes[0], color='#1f77b4')
        axes[0].set_xlabel('총 조회수', fontweight='bold')
        axes[0].set_title('채널별 총 조회수 TOP10', fontweight='bold')
        
        # 좋아요
        channel_stats['like_count'].plot(kind='barh', ax=axes[1], color='#ff7f0e')
        axes[1].set_xlabel('총 좋아요', fontweight='bold')
        axes[1].set_title('채널별 총 좋아요 TOP10', fontweight='bold')
        
        # 댓글
        channel_stats['comment_count'].plot(kind='barh', ax=axes[2], color='#2ca02c')
        axes[2].set_xlabel('총 댓글', fontweight='bold')
        axes[2].set_title('채널별 총 댓글 TOP10', fontweight='bold')
        
        plt.tight_layout()
        return fig
    
    def plot_upload_trend(self, figsize=(14, 6)):
        """날짜별 업로드 추세"""
        daily_uploads = self.df.groupby(self.df['upload_date'].dt.date).size()
        daily_views = self.df.groupby(self.df['upload_date'].dt.date)['view_count'].sum()
        
        fig, ax1 = plt.subplots(figsize=figsize)
        
        # 업로드 개수
        ax1.plot(daily_uploads.index, daily_uploads.values, color='#1f77b4', 
                marker='o', linewidth=2, label='업로드 개수', markersize=4)
        ax1.set_xlabel('날짜', fontweight='bold')
        ax1.set_ylabel('업로드 개수', color='#1f77b4', fontweight='bold')
        ax1.tick_params(axis='y', labelcolor='#1f77b4')
        
        # 조회수
        ax2 = ax1.twinx()
        ax2.plot(daily_views.index, daily_views.values, color='#ff7f0e', 
                marker='s', linewidth=2, label='조회수', markersize=4)
        ax2.set_ylabel('조회수', color='#ff7f0e', fontweight='bold')
        ax2.tick_params(axis='y', labelcolor='#ff7f0e')
        
        plt.title('날짜별 업로드 추세 및 조회수', fontsize=14, fontweight='bold', pad=20)
        plt.xticks(rotation=45, ha='right')
        
        # 범례
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        plt.tight_layout()
        return fig
    
    def plot_weekday_distribution(self, figsize=(12, 6)):
        """요일별 업로드 분포"""
        weekday_names = ['월', '화', '수', '목', '금', '토', '일']
        weekday_counts = self.df['upload_date'].dt.dayofweek.value_counts().sort_index()
        
        fig, ax = plt.subplots(figsize=figsize)
        colors = sns.color_palette("husl", 7)
        bars = ax.bar(range(7), [weekday_counts.get(i, 0) for i in range(7)], color=colors)
        
        ax.set_xticks(range(7))
        ax.set_xticklabels(weekday_names, fontsize=11)
        ax.set_ylabel('업로드 개수', fontweight='bold')
        ax.set_title('요일별 업로드 분포', fontsize=14, fontweight='bold', pad=20)
        
        # 값 표시
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        return fig
    
    def plot_scatter_views_vs_likes(self, figsize=(10, 7)):
        """조회수 vs 좋아요 산점도"""
        fig, ax = plt.subplots(figsize=figsize)
        
        scatter = ax.scatter(self.df['view_count'], self.df['like_count'], 
                           alpha=0.6, s=100, c=self.df['comment_count'], 
                           cmap='viridis', edgecolors='black', linewidth=0.5)
        
        ax.set_xlabel('조회수', fontweight='bold', fontsize=12)
        ax.set_ylabel('좋아요 수', fontweight='bold', fontsize=12)
        ax.set_title('조회수 vs 좋아요 (색상: 댓글 수)', fontsize=14, fontweight='bold', pad=20)
        
        # 상관계수
        correlation = self.df['view_count'].corr(self.df['like_count'])
        ax.text(0.05, 0.95, f'상관계수: {correlation:.3f}', 
               transform=ax.transAxes, fontsize=11, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('댓글 수', fontweight='bold')
        
        plt.tight_layout()
        return fig
    
    def plot_scatter_views_vs_comments(self, figsize=(10, 7)):
        """조회수 vs 댓글 산점도"""
        fig, ax = plt.subplots(figsize=figsize)
        
        scatter = ax.scatter(self.df['view_count'], self.df['comment_count'], 
                           alpha=0.6, s=100, c=self.df['like_count'], 
                           cmap='plasma', edgecolors='black', linewidth=0.5)
        
        ax.set_xlabel('조회수', fontweight='bold', fontsize=12)
        ax.set_ylabel('댓글 수', fontweight='bold', fontsize=12)
        ax.set_title('조회수 vs 댓글 (색상: 좋아요 수)', fontsize=14, fontweight='bold', pad=20)
        
        # 상관계수
        correlation = self.df['view_count'].corr(self.df['comment_count'])
        ax.text(0.05, 0.95, f'상관계수: {correlation:.3f}', 
               transform=ax.transAxes, fontsize=11, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('좋아요 수', fontweight='bold')
        
        plt.tight_layout()
        return fig
    
    def plot_views_distribution(self, figsize=(12, 6)):
        """조회수 분포"""
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # 히스토그램
        axes[0].hist(self.df['view_count'], bins=50, color='#1f77b4', edgecolor='black', alpha=0.7)
        axes[0].set_xlabel('조회수', fontweight='bold')
        axes[0].set_ylabel('동영상 개수', fontweight='bold')
        axes[0].set_title('조회수 분포 (히스토그램)', fontweight='bold')
        
        # 박스플롯
        axes[1].boxplot([self.df['view_count'], self.df['like_count'], self.df['comment_count']],
                       labels=['조회수', '좋아요', '댓글'])
        axes[1].set_ylabel('개수', fontweight='bold')
        axes[1].set_title('조회수, 좋아요, 댓글 분포 (박스플롯)', fontweight='bold')
        axes[1].set_yscale('log')
        
        plt.tight_layout()
        return fig
    
    def plot_title_wordcloud(self, figsize=(14, 8)):
        """제목 워드클라우드"""
        all_titles = ' '.join(self.df['title'].dropna().astype(str))
        
        wordcloud = WordCloud(
            width=figsize[0]*100,
            height=figsize[1]*100,
            background_color='white',
            colormap='viridis',
            max_words=100,
            relative_scaling=0.5,
            min_font_size=10
        ).generate(all_titles)
        
        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        ax.set_title('제목 워드클라우드', fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        return fig
    
    def plot_tags_wordcloud(self, figsize=(14, 8)):
        """태그 워드클라우드"""
        all_tags = ' '.join(self.df['tags'].dropna().astype(str))
        
        wordcloud = WordCloud(
            width=figsize[0]*100,
            height=figsize[1]*100,
            background_color='white',
            colormap='plasma',
            max_words=100,
            relative_scaling=0.5,
            min_font_size=10
        ).generate(all_tags)
        
        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        ax.set_title('태그 워드클라우드', fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        return fig
    
    def plot_engagement_ratio(self, figsize=(12, 6)):
        """참여도 비율 분석 (좋아요/조회수, 댓글/조회수)"""
        self.df['like_ratio'] = (self.df['like_count'] / self.df['view_count'] * 100).replace([np.inf, -np.inf], 0)
        self.df['comment_ratio'] = (self.df['comment_count'] / self.df['view_count'] * 100).replace([np.inf, -np.inf], 0)
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # 좋아요 비율
        axes[0].hist(self.df['like_ratio'], bins=50, color='#ff7f0e', edgecolor='black', alpha=0.7)
        axes[0].set_xlabel('좋아요 비율 (%)', fontweight='bold')
        axes[0].set_ylabel('동영상 개수', fontweight='bold')
        axes[0].set_title(f'좋아요 비율 분포\n평균: {self.df["like_ratio"].mean():.2f}%', fontweight='bold')
        
        # 댓글 비율
        axes[1].hist(self.df['comment_ratio'], bins=50, color='#2ca02c', edgecolor='black', alpha=0.7)
        axes[1].set_xlabel('댓글 비율 (%)', fontweight='bold')
        axes[1].set_ylabel('동영상 개수', fontweight='bold')
        axes[1].set_title(f'댓글 비율 분포\n평균: {self.df["comment_ratio"].mean():.2f}%', fontweight='bold')
        
        plt.tight_layout()
        return fig
    
    def create_interactive_dashboard(self, output_file='interactive_dashboard.html'):
        """Plotly를 사용한 인터랙티브 대시보드 생성"""
        
        # 1. 상위 동영상 (조회수)
        top_videos = self.df.nlargest(10, 'view_count')
        fig1 = go.Figure(data=[
            go.Bar(y=top_videos['title'].str[:40], x=top_videos['view_count'], 
                   orientation='h', marker=dict(color='#1f77b4'))
        ])
        fig1.update_layout(title='상위 10개 인기 동영상', xaxis_title='조회수', height=500)
        
        # 2. 채널별 동영상 수
        channel_counts = self.df['channel_name'].value_counts().head(10)
        fig2 = go.Figure(data=[
            go.Bar(x=channel_counts.index, y=channel_counts.values, marker=dict(color='#ff7f0e'))
        ])
        fig2.update_layout(title='상위 10개 채널 (동영상 개수)', xaxis_title='채널', 
                          yaxis_title='동영상 개수', xaxis_tickangle=-45, height=500)
        
        # 3. 조회수 vs 좋아요
        fig3 = px.scatter(self.df, x='view_count', y='like_count', 
                         color='comment_count', hover_name='title',
                         title='조회수 vs 좋아요', labels={'view_count': '조회수', 'like_count': '좋아요 수'})
        fig3.update_layout(height=500)
        
        # 4. 날짜별 추세
        daily_uploads = self.df.groupby(self.df['upload_date'].dt.date).size()
        fig4 = go.Figure(data=[
            go.Scatter(x=daily_uploads.index, y=daily_uploads.values, 
                      mode='lines+markers', name='업로드 개수')
        ])
        fig4.update_layout(title='날짜별 업로드 추세', xaxis_title='날짜', 
                          yaxis_title='업로드 개수', height=500)
        
        # 5. 요일별 분포
        weekday_names = ['월', '화', '수', '목', '금', '토', '일']
        weekday_counts = self.df['upload_date'].dt.dayofweek.value_counts().sort_index()
        fig5 = go.Figure(data=[
            go.Bar(x=weekday_names, y=[weekday_counts.get(i, 0) for i in range(7)],
                  marker=dict(color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']))
        ])
        fig5.update_layout(title='요일별 업로드 분포', xaxis_title='요일', 
                          yaxis_title='업로드 개수', height=500)
        
        # HTML 파일에 모두 저장
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('<html><head><meta charset="utf-8"></head><body>')
            f.write('<h1 style="text-align:center">YouTube 데이터 분석 대시보드</h1>')
            f.write('<hr>')
            f.write(fig1.to_html(include_plotlyjs='cdn', div_id='fig1'))
            f.write(fig2.to_html(include_plotlyjs=False, div_id='fig2'))
            f.write(fig3.to_html(include_plotlyjs=False, div_id='fig3'))
            f.write(fig4.to_html(include_plotlyjs=False, div_id='fig4'))
            f.write(fig5.to_html(include_plotlyjs=False, div_id='fig5'))
            f.write('</body></html>')
        
        print(f"\n✓ 인터랙티브 대시보드 저장: {output_file}")
    
    def generate_full_report(self, output_dir='analysis_report'):
        """모든 차트를 저장하고 HTML 리포트 생성"""
        import os
        
        # 디렉토리 생성
        os.makedirs(output_dir, exist_ok=True)
        
        print("\n" + "="*50)
        print(" 분석 리포트 생성 중...")
        print("="*50)
        
        # 기본 통계 출력
        self.print_summary_statistics()
        
        # 모든 차트 생성 및 저장
        charts = {
            '01_top_videos.png': self.plot_top_videos(),
            '02_top_channels.png': self.plot_top_channels(),
            '03_channel_statistics.png': self.plot_channel_statistics(),
            '04_upload_trend.png': self.plot_upload_trend(),
            '05_weekday_distribution.png': self.plot_weekday_distribution(),
            '06_views_distribution.png': self.plot_views_distribution(),
            '07_scatter_views_vs_likes.png': self.plot_scatter_views_vs_likes(),
            '08_scatter_views_vs_comments.png': self.plot_scatter_views_vs_comments(),
            '09_engagement_ratio.png': self.plot_engagement_ratio(),
            '10_title_wordcloud.png': self.plot_title_wordcloud(),
            '11_tags_wordcloud.png': self.plot_tags_wordcloud(),
        }
        
        for filename, fig in charts.items():
            filepath = os.path.join(output_dir, filename)
            fig.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f" {filename} 저장")
            plt.close(fig)
        
        # 인터랙티브 대시보드
        self.create_interactive_dashboard(os.path.join(output_dir, 'interactive_dashboard.html'))
        
        # HTML 리포트 생성
        self._generate_html_report(output_dir)
        
        print("\n" + "="*50)
        print(f"분석 리포트 생성 완료: {output_dir}")
        print("="*50)

    def _generate_html_report(self, output_dir):
        """HTML 리포트 생성"""
        stats = self.get_summary_statistics()
        html_content = f"""
<!DOCTYPE html><html lang="ko"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0"><title>YouTube 데이터 분석 리포트</title><style>body{{font-family:'Arial',sans-serif;margin:20px;background-color:#f5f5f5;color:#333;}}.container{{max-width:1200px;margin:0 auto;background-color:#fff;padding:30px;border-radius:10px;box-shadow:0 2px 10px rgba(0,0,0,0.1);}}h1{{text-align:center;color:#1f77b4;border-bottom:3px solid #1f77b4;padding-bottom:15px;}}h2{{color:#ff7f0e;margin-top:30px;border-left:5px solid #ff7f0e;padding-left:10px;}}.stats-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:20px;margin:20px 0;}}.stat-card{{background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);color:#fff;padding:20px;border-radius:8px;box-shadow:0 4px 6px rgba(0,0,0,0.1);}}.stat-card h3{{margin:0 0 10px 0;font-size:14px;opacity:.9;}}.stat-card .value{{font-size:24px;font-weight:bold;}}.chart-section{{margin:40px 0;padding:20px;background-color:#f9f9f9;border-radius:8px;}}.chart-section img{{max-width:100%;height:auto;border-radius:5px;}}.link-section{{text-align:center;margin:30px 0;}}.link-section a{{display:inline-block;padding:12px 24px;margin:10px;background-color:#1f77b4;color:#fff;text-decoration:none;border-radius:5px;transition:background-color .3s;}}.link-section a:hover{{background-color:#ff7f0e;}}footer{{text-align:center;margin-top:40px;padding-top:20px;border-top:1px solid #ddd;color:#666;font-size:12px;}}</style></head><body><div class="container"><h1>YouTube 데이터 분석 리포트</h1><h2>기본 통계</h2><div class="stats-grid"><div class="stat-card"><h3>총 동영상 수</h3><div class="value">{stats['총 동영상 수']}</div></div><div class="stat-card"><h3>채널 수</h3><div class="value">{stats['채널 수']}</div></div><div class="stat-card"><h3>총 조회수</h3><div class="value">{stats['총 조회수']}</div></div><div class="stat-card"><h3>총 좋아요</h3><div class="value">{stats['총 좋아요']}</div></div><div class="stat-card"><h3>평균 조회수</h3><div class="value">{stats['평균 조회수']}</div></div><div class="stat-card"><h3>평균 좋아요</h3><div class="value">{stats['평균 좋아요']}</div></div></div><h2>분석 결과</h2><div class="chart-section"><h3>상위 10개 인기 동영상 (조회수 기준)</h3><img src="01_top_videos.png" alt="상위 인기 동영상"></div><div class="chart-section"><h3>상위 10개 채널 (동영상 개수)</h3><img src="02_top_channels.png" alt="상위 채널"></div><div class="chart-section"><h3>채널별 통계</h3><img src="03_channel_statistics.png" alt="채널별 통계"></div><div class="chart-section"><h3>날짜별 업로드 추세</h3><img src="04_upload_trend.png" alt="업로드 추세"></div><div class="chart-section"><h3>요일별 업로드 분포</h3><img src="05_weekday_distribution.png" alt="요일별 분포"></div><div class="chart-section"><h3>조회수, 좋아요, 댓글 분포</h3><img src="06_views_distribution.png" alt="분포 분석"></div><div class="chart-section"><h3>조회수 vs 좋아요 상관관계</h3><img src="07_scatter_views_vs_likes.png" alt="조회수 vs 좋아요"></div><div class="chart-section"><h3>조회수 vs 댓글 상관관계</h3><img src="08_scatter_views_vs_comments.png" alt="조회수 vs 댓글"></div><div class="chart-section"><h3>참여도 비율 분석</h3><img src="09_engagement_ratio.png" alt="참여도 비율"></div><div class="chart-section"><h3>제목 워드클라우드</h3><img src="10_title_wordcloud.png" alt="제목 워드클라우드"></div><div class="chart-section"><h3>태그 워드클라우드</h3><img src="11_tags_wordcloud.png" alt="태그 워드클라우드"></div><div class="link-section"><a href="interactive_dashboard.html" target="_blank">인터랙티브 대시보드 보기</a></div><footer><p>이 리포트는 YouTube 크롤링 데이터 분석 도구로 생성되었습니다.</p><p>생성 시간: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p></footer></div></body></html>
"""
        report_path = os.path.join(output_dir, 'report.html')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        print(f"HTML 리포트 저장: report.html")


class CommentAnalyzer:
    """댓글 데이터 분석 클래스"""
    
    def __init__(self, csv_file):
        """
        댓글 분석기 초기화
        
        Args:
            csv_file: 댓글 데이터 CSV 파일 경로
        """
        self.df = pd.read_csv(csv_file, encoding='utf-8-sig')
        print(f"✓ {len(self.df)}개의 댓글 데이터 로드 완료")
    
    def get_summary_statistics(self):
        """댓글 기본 통계"""
        stats = {
            '총 댓글 수': len(self.df),
            '고유 사용자 수': self.df['author'].nunique() if 'author' in self.df.columns else 'N/A',
            '평균 댓글 길이': f"{self.df['text'].str.len().mean():.0f} 자" if 'text' in self.df.columns else 'N/A',
        }
        return stats
    
    def print_summary_statistics(self):
        """댓글 통계 출력"""
        stats = self.get_summary_statistics()
        print("\n" + "="*50)
        print(" 댓글 분석 통계")
        print("="*50)
        for key, value in stats.items():
            print(f"{key:.<30} {value}")
    
    def plot_comment_wordcloud(self, figsize=(14, 8)):
        """댓글 워드클라우드"""
        if 'text' not in self.df.columns:
            print("'text' 컬럼이 없습니다.")
            return None
        
        all_comments = ' '.join(self.df['text'].dropna().astype(str))
        
        wordcloud = WordCloud(
            width=figsize[0]*100,
            height=figsize[1]*100,
            background_color='white',
            colormap='coolwarm',
            max_words=100,
            relative_scaling=0.5,
            min_font_size=10
        ).generate(all_comments)
        
        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        ax.set_title('댓글 워드클라우드', fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        return fig


def analyze_youtube_data(csv_file, output_dir='analysis_report'):
    """YouTube 데이터 분석 메인 함수"""
    analyzer = YouTubeAnalyzer(csv_file)
    analyzer.generate_full_report(output_dir)
    return analyzer


def analyze_comments_data(csv_file):
    """댓글 데이터 분석 메인 함수"""
    analyzer = CommentAnalyzer(csv_file)
    analyzer.print_summary_statistics()
    return analyzer


# ---------------------------------------------------------
# YouTube Crawler Code
# ---------------------------------------------------------

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
        api_key = 'AQ.Ab8RN6JxKF3ipcxM0vufB_SLkGNeWAGDJtFQUT7obH8G1o-R4A'
        
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

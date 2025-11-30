"""
YouTube 크롤링 및 데이터 분석 통합 프로그램
"""

import os
import re
import sys
import pickle
import warnings
from datetime import datetime
from typing import List, Tuple, Set

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from matplotlib import font_manager, rc
import plotly.graph_objects as go
import plotly.express as px
from dotenv import load_dotenv
from konlpy.tag import Okt
from deep_translator import GoogleTranslator
import nltk
from nltk.corpus import stopwords
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

import sys
import subprocess
import pkg_resources

# git hub 링크 : https://github.com/SbEnChOi/Crolling_assistant.git


# ---------------------------------------------------------
# Embedded Configurations
# ---------------------------------------------------------

REQUIRED_PACKAGES = [
    'python-dotenv==1.0.1',
    'openpyxl==3.1.2',
    'pandas==2.3.3',
    'numpy==2.2.6',
    'matplotlib==3.10.6',
    'seaborn==0.13.2',
    'wordcloud==1.9.4',
    'plotly==5.23.0',
    'konlpy==0.6.0',
    'deep-translator==1.11.4',
    'nltk==3.9.2',
    'google-api-python-client==2.176.0',
    'google-auth==2.28.0',
    'google-auth-oauthlib==1.2.0',
    'google-auth-httplib2==0.2.0',
    'requests==2.32.4',
    'beautifulsoup4==4.13.4',
    'lxml==6.0.2'
]

CLIENT_CONFIG = {
    "installed": {
        "client_id": "616770459881-f5ip9jvluj0gdkm9443p04tnn6b1m0dk.apps.googleusercontent.com",
        "project_id": "gen-lang-client-0741447260",
        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
        "token_uri": "https://oauth2.googleapis.com/token",
        "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
        "client_secret": "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",
        "redirect_uris": ["http://localhost"]
    }
}

def install_requirements():
    # 설치가 필요한 패키지 리스트 확인
    missing_packages = []
    for package in REQUIRED_PACKAGES:
        try:
            pkg_resources.require(package)
        except (pkg_resources.DistributionNotFound, pkg_resources.VersionConflict):
            missing_packages.append(package)

    # 누락된 패키지가 있을 경우에만 설치 실행
    if missing_packages:
        print(f"설치되지 않은 패키지 발견: {missing_packages}")
        print("패키지 설치를 시작합니다...")
        python = sys.executable
        subprocess.check_call([python, '-m', 'pip', 'install', *missing_packages])
        print("설치 완료!")
    else:
        print("모든 패키지가 이미 설치되어 있습니다.")


#-----------------------------------------
# 키워드 추출 부분
#-----------------------------------------


# NLTK 데이터 다운로드
for resource in ['tokenizers/punkt', 'corpora/stopwords']:
    try:
        nltk.data.find(resource)
    except LookupError:
        nltk.download(resource.split('/')[-1], quiet=True)

try:
    ENGLISH_STOPWORDS = set(stopwords.words('english'))
except Exception:
    ENGLISH_STOPWORDS = {'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}

# 한국어 불용어 및 조사 목록
KOREAN_STOPWORDS = {
    '이', '가', '을', '를', '에', '에서', '으로', '로', '의', '과', '와', '도', '만', '부터', '까지',
    '는', '은', '에게', '께', '한테', '더', '또', '그', '그것', '이것', '저것', '그런', '이런', '저런',
    '그리고', '그러나', '하지만', '또한', '그래서', '따라서', '그런데', '그럼', '그렇다면',
    '있다', '없다', '하다', '되다', '이다', '아니다', '것', '수', '때', '곳', '거', '게',
    '등', '등등', '및', '또는', '혹은', '만약', '만일',
    '에게서', '한테서', '께서', '처럼', '만큼', '보다', '같이', '커녕', '마저', '조차',
    '든지', '이나', '든가', '라도', '이라도', '부터는', '부터도'
}

JOSA_PATTERNS = [
    r'이$', r'가$', r'을$', r'를$', r'에$', r'에서$', r'으로$', r'로$', r'의$', r'과$', r'와$',
    r'는$', r'은$', r'도$', r'만$', r'부터$', r'까지$', r'에게$', r'께$', r'한테$',
    r'에서$', r'으로부터$', r'처럼$', r'만큼$', r'보다$', r'같이$', r'마저$', r'조차$',
    r'든지$', r'이나$', r'든가$', r'라도$', r'이라도$', r'부터는$', r'부터도$'
]

class KeywordExtractor:
    def __init__(self):
        self.okt = self._safe_init(Okt)
        self.translator = self._safe_init(lambda: GoogleTranslator(source='ko', target='en'))
    
    @staticmethod
    def _safe_init(func): 
        try: return func()
        except: return None
    
    def extract_keywords(self, sentence: str, min_length: int = 2) -> List[str]:
        if not (sentence := (sentence or '').strip()): return []
        words = sentence.split()
        if len(words) <= 2: return [sentence.replace(' ', '')] if sentence.replace(' ', '') else []
        if self.okt is None: return [w for w in words if w not in KOREAN_STOPWORDS and len(w) >= min_length and not w.isdigit()] or [sentence]
        try:
            keywords = [self._remove_josa(w) for w, pos in self.okt.pos(sentence, norm=True, stem=True) 
                       if not pos.startswith('J') and pos[0] in 'NVA' and w not in KOREAN_STOPWORDS and len(w) >= min_length and not w.isdigit()]
            return sorted(list(set(k for k in keywords if k and len(k) >= min_length)), key=len, reverse=True) or [sentence]
        except: return [sentence]
    
    def _remove_josa(self, word: str) -> str:
        if not word or word in KOREAN_STOPWORDS: return ''
        cleaned = word
        for _ in range(3):
            prev = cleaned
            for p in JOSA_PATTERNS: cleaned = re.sub(p, '', cleaned)
            if cleaned == prev: break
        return cleaned if cleaned and len(cleaned) >= 2 and cleaned not in KOREAN_STOPWORDS else ''
    
    def translate_keywords(self, keywords: List[str]) -> List[str]:
        if not self.translator: return [kw.lower() for kw in keywords if self._is_english(kw)]
        result = []
        for kw in keywords:
            if self._is_english(kw): result.append(kw.lower())
            else:
                try: 
                    t = self.translator.translate(kw)
                    if t and t != kw: result.append(t.lower())
                except: pass
        return result
    
    def _is_english(self, text: str) -> bool:
        try: 
            text.encode('ascii'); return True
        except: return False
    
    def filter_english_keywords(self, keywords: List[str]) -> List[str]:
        filtered = [' '.join(w for w in kw.split() if w.lower() not in ENGLISH_STOPWORDS and len(w) >= 2) for kw in keywords]
        return [f for f in filtered if f] or keywords

def select_keywords(keywords: List[str]) -> List[str]:
    if not keywords:
        return []
    
    print("\n추출된 키워드 목록:")
    for keyword in keywords:
        print(f"  - {keyword}")
    
    while True:
        try:
            selection = input("\n사용할 키워드 입력 (콤마 구분, 'all' 전체): ").strip()
            if selection.lower() == 'all':
                return keywords
            
            if not selection:
                continue
            
            input_keywords = [kw.strip() for kw in selection.split(',') if kw.strip()]
            if not input_keywords:
                continue
            
            selected_keywords = []
            for input_kw in input_keywords:
                for kw in keywords:
                    if kw.lower() == input_kw.lower() or kw == input_kw:
                        selected_keywords.append(kw)
                        break
            
            if selected_keywords:
                unique_selected = list(dict.fromkeys(selected_keywords))
                print(f"선택된 키워드: {', '.join(unique_selected)}")
                return unique_selected
            print("선택된 키워드가 없습니다.")
                
        except KeyboardInterrupt:
            return []
        except Exception:
            return keywords

def extract_and_translate_keywords(sentence: str) -> Tuple[List[str], List[str]]:
    extractor = KeywordExtractor()
    korean_keywords = extractor.extract_keywords(sentence)
    selected_korean_keywords = select_keywords(korean_keywords)
    
    if not selected_korean_keywords:
        selected_korean_keywords = [sentence]
    
    english_keywords = extractor.translate_keywords(selected_korean_keywords)
    english_keywords = extractor.filter_english_keywords(english_keywords)
    
    return selected_korean_keywords, english_keywords

warnings.filterwarnings('ignore')
sns.set_style("whitegrid")
sns.set_palette("husl")
# 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

class YouTubeAnalyzer:
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file, encoding='utf-8-sig')
        self.df['upload_date'] = pd.to_datetime(self.df['upload_date'], errors='coerce')
        for col in ['view_count', 'like_count', 'comment_count']:
            self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
        
        print(f"✓ {len(self.df)}개 데이터 로드 ({self.df['upload_date'].min()} ~ {self.df['upload_date'].max()})")
    
    def get_summary_statistics(self):
        return {
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
    
    def print_summary_statistics(self):
        stats = self.get_summary_statistics()
        print("\n" + "="*50 + "\n기본 통계\n" + "="*50)
        for key, value in stats.items():
            print(f"{key:.<30} {value}")
    
    def plot_top_videos(self, n=10, figsize=(14, 8)):
        top_videos = self.df.nlargest(n, 'view_count')[['title', 'view_count']]
        top_videos['title'] = top_videos['title'].str[:50] + '...'
        
        fig, ax = plt.subplots(figsize=figsize)
        y_pos = np.arange(len(top_videos))
        ax.barh(y_pos, top_videos['view_count'].values, color='#1f77b4')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_videos['title'].values)
        ax.set_title(f'상위 {n}개 인기 동영상 (조회수)', fontsize=14, fontweight='bold')
        ax.invert_yaxis()
        
        for i, v in enumerate(top_videos['view_count'].values):
            ax.text(v, i, f' {int(v):,}', va='center')
        
        plt.tight_layout()
        return fig
    
    def plot_top_channels(self, n=10, figsize=(12, 6)):
        top_channels = self.df['channel_name'].value_counts().head(n)
        fig, ax = plt.subplots(figsize=figsize)
        bars = ax.bar(range(len(top_channels)), top_channels.values)
        ax.set_xticks(range(len(top_channels)))
        ax.set_xticklabels(top_channels.index, rotation=45, ha='right')
        ax.set_title(f'상위 {n}개 채널 (동영상 수)', fontsize=14, fontweight='bold')
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height, f'{int(height)}', ha='center', va='bottom')
        
        plt.tight_layout()
        return fig

    def plot_channel_statistics(self, n=10, figsize=(14, 8)):
        channel_stats = self.df.groupby('channel_name').agg({
            'view_count': 'sum', 'like_count': 'sum', 'comment_count': 'sum'
        }).nlargest(n, 'view_count')
        
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        for i, (col, color) in enumerate(zip(['view_count', 'like_count', 'comment_count'], ['#1f77b4', '#ff7f0e', '#2ca02c'])):
            channel_stats[col].plot(kind='barh', ax=axes[i], color=color)
            axes[i].set_title(f'채널별 {col} TOP10')
        
        plt.tight_layout()
        return fig

    def plot_upload_trend(self, figsize=(14, 6)):
        daily_uploads = self.df.groupby(self.df['upload_date'].dt.date).size()
        daily_views = self.df.groupby(self.df['upload_date'].dt.date)['view_count'].sum()
        
        fig, ax1 = plt.subplots(figsize=figsize)
        ax1.plot(daily_uploads.index, daily_uploads.values, color='#1f77b4', marker='o', label='업로드')
        ax1.set_ylabel('업로드 수', color='#1f77b4')
        
        ax2 = ax1.twinx()
        ax2.plot(daily_views.index, daily_views.values, color='#ff7f0e', marker='s', label='조회수')
        ax2.set_ylabel('조회수', color='#ff7f0e')
        
        plt.title('날짜별 업로드 및 조회수 추세')
        plt.tight_layout()
        return fig

    def plot_weekday_distribution(self, figsize=(12, 6)):
        weekday_counts = self.df['upload_date'].dt.dayofweek.value_counts().sort_index()
        fig, ax = plt.subplots(figsize=figsize)
        bars = ax.bar(range(7), [weekday_counts.get(i, 0) for i in range(7)])
        ax.set_xticklabels(['월', '화', '수', '목', '금', '토', '일'])
        ax.set_title('요일별 업로드 분포')
        
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(), f'{int(bar.get_height())}', ha='center', va='bottom')
        plt.tight_layout()
        return fig

    def plot_scatter_views_vs_likes(self, figsize=(10, 7)):
        fig, ax = plt.subplots(figsize=figsize)
        scatter = ax.scatter(self.df['view_count'], self.df['like_count'], c=self.df['comment_count'], cmap='viridis', alpha=0.6)
        ax.set_xlabel('조회수')
        ax.set_ylabel('좋아요')
        ax.set_title(f'조회수 vs 좋아요 (상관관계: {self.df["view_count"].corr(self.df["like_count"]):.3f})')
        plt.colorbar(scatter, label='댓글 수')
        plt.tight_layout()
        return fig

    def plot_scatter_views_vs_comments(self, figsize=(10, 7)):
        fig, ax = plt.subplots(figsize=figsize)
        scatter = ax.scatter(self.df['view_count'], self.df['comment_count'], c=self.df['like_count'], cmap='plasma', alpha=0.6)
        ax.set_xlabel('조회수')
        ax.set_ylabel('댓글')
        ax.set_title(f'조회수 vs 댓글 (상관관계: {self.df["view_count"].corr(self.df["comment_count"]):.3f})')
        plt.colorbar(scatter, label='좋아요 수')
        plt.tight_layout()
        return fig

    def plot_views_distribution(self, figsize=(12, 6)):
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        axes[0].hist(self.df['view_count'], bins=50, color='#1f77b4')
        axes[0].set_title('조회수 분포')
        
        axes[1].boxplot([self.df['view_count'], self.df['like_count'], self.df['comment_count']], labels=['조회수', '좋아요', '댓글'])
        axes[1].set_yscale('log')
        axes[1].set_title('분포 (Log Scale)')
        plt.tight_layout()
        return fig

    def plot_engagement_ratio(self, figsize=(12, 6)):
        self.df['like_ratio'] = (self.df['like_count'] / self.df['view_count'] * 100).fillna(0)
        self.df['comment_ratio'] = (self.df['comment_count'] / self.df['view_count'] * 100).fillna(0)
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        axes[0].hist(self.df['like_ratio'], bins=50, color='#ff7f0e')
        axes[0].set_title(f'좋아요 비율 (평균: {self.df["like_ratio"].mean():.2f}%)')
        
        axes[1].hist(self.df['comment_ratio'], bins=50, color='#2ca02c')
        axes[1].set_title(f'댓글 비율 (평균: {self.df["comment_ratio"].mean():.2f}%)')
        
        plt.tight_layout()
        return fig

    def plot_title_wordcloud(self, figsize=(14, 8)):
        text = ' '.join(self.df['title'].dropna().astype(str))
        font_path = font_manager.findfont('Malgun Gothic')
        wc = WordCloud(width=figsize[0]*100, height=figsize[1]*100, background_color='white', font_path=font_path).generate(text)
        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(wc, interpolation='bilinear')
        ax.axis('off')
        ax.set_title('제목 워드클라우드')
        plt.tight_layout()
        return fig

    def plot_tags_wordcloud(self, figsize=(14, 8)):
        text = ' '.join(self.df['tags'].dropna().astype(str))
        font_path = font_manager.findfont('Malgun Gothic')
        wc = WordCloud(width=figsize[0]*100, height=figsize[1]*100, background_color='white', colormap='plasma', font_path=font_path).generate(text)
        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(wc, interpolation='bilinear')
        ax.axis('off')
        ax.set_title('태그 워드클라우드')
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

    def generate_full_report(self, output_dir='analysis_report'):
        os.makedirs(output_dir, exist_ok=True)
        print("\n분석 리포트 생성 중...")
        self.print_summary_statistics()
        
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
            if fig:
                fig.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
                plt.close(fig)
        
        # 인터랙티브 대시보드
        self.create_interactive_dashboard(os.path.join(output_dir, 'interactive_dashboard.html'))
        
        # HTML 리포트 생성
        self._generate_html_report(output_dir)
        
        print(f"리포트 생성 완료: {output_dir}")

class CommentAnalyzer:
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file, encoding='utf-8-sig')
        print(f"✓ {len(self.df)}개 댓글 데이터 로드")
    
    def print_summary_statistics(self):
        print(f"\n총 댓글 수: {len(self.df)}")
        if 'author' in self.df.columns:
            print(f"고유 사용자: {self.df['author'].nunique()}")

def analyze_youtube_data(csv_file, output_dir='analysis_report'):
    analyzer = YouTubeAnalyzer(csv_file)
    analyzer.generate_full_report(output_dir)
    return analyzer

def analyze_comments_data(csv_file):
    analyzer = CommentAnalyzer(csv_file)
    analyzer.print_summary_statistics()
    return analyzer

# ---------------------------------------------------------
# YouTube Crawler Code
# ---------------------------------------------------------

load_dotenv('key.env')
load_dotenv('.env')

SCOPES = ['https://www.googleapis.com/auth/youtube.force-ssl']
TOKEN_FILE = 'youtube_token.pickle'

def sanitize_filename(text):
    text = re.sub(r'[^\w\s-]', '', text)
    return re.sub(r'[-\s]+', '_', text)

def get_user_input():
    print("=" * 50 + "\nYouTube 크롤링 프로그램\n" + "=" * 50)
    
    search_sentence = input("크롤링할 내용(문장): ").strip()
    if not search_sentence:
        raise ValueError("내용을 입력해주세요.")
    
    print("\n키워드 추출 중...")
    korean_keywords, english_keywords = extract_and_translate_keywords(search_sentence)
    all_keywords = list(set(korean_keywords + english_keywords))
    
    print(f"키워드: {', '.join(all_keywords)}")
    
    start_date = datetime.strptime(input("시작 날짜 (YYYY-MM-DD): ").strip(), "%Y-%m-%d")
    end_date = datetime.strptime(input("끝 날짜 (YYYY-MM-DD): ").strip(), "%Y-%m-%d")
    
    if start_date > end_date:
        raise ValueError("날짜 범위 오류")
    
    max_videos = int(input("크롤링할 동영상 개수: ").strip())
    collect_comments = input("댓글 수집? (y/n): ").strip().lower() in ('y', 'yes')
    
    sort_choice = input("정렬 (1:관련도, 2:최신, 3:조회수, 4:평점) [1]: ").strip()
    order = {'1': 'relevance', '2': 'date', '3': 'viewCount', '4': 'rating'}.get(sort_choice, 'relevance')
    
    return all_keywords, start_date, end_date, max_videos, collect_comments, order

def authenticate_oauth():
    credentials = None
    if os.path.exists(TOKEN_FILE):
        try:
            with open(TOKEN_FILE, 'rb') as token:
                credentials = pickle.load(token)
        except Exception:
            pass
    
    if not credentials or not credentials.valid:
        if credentials and credentials.expired and credentials.refresh_token:
            credentials.refresh(Request())
        else:
            # flow = InstalledAppFlow.from_client_secrets_file('client_secret.json', SCOPES)
            flow = InstalledAppFlow.from_client_config(CLIENT_CONFIG, SCOPES)
            credentials = flow.run_local_server(port=0)
        
        with open(TOKEN_FILE, 'wb') as token:
            pickle.dump(credentials, token)
    
    return build('youtube', 'v3', credentials=credentials)

def connect_youtube():
    try:
        return authenticate_oauth()
    except Exception as e:
        print(f"OAuth 실패 ({e}), API 키 사용 시도...")
        api_key = os.getenv('YOUTUBE_API_KEY')
        if not api_key:
            raise ValueError("API 키 없음")
        return build('youtube', 'v3', developerKey=api_key)

def search_videos(youtube, keyword, start_date, end_date, order='relevance', max_results=50):
    try:
        published_after = start_date.strftime('%Y-%m-%dT%H:%M:%SZ')
        published_before = (end_date.replace(hour=23, minute=59, second=59)).strftime('%Y-%m-%dT%H:%M:%SZ')
        
        response = youtube.search().list(
            q=keyword, part='id,snippet', type='video', order=order,
            publishedAfter=published_after, publishedBefore=published_before,
            maxResults=min(max_results, 50), relevanceLanguage='en'
        ).execute()
        
        return [item['id']['videoId'] for item in response.get('items', [])]
    except HttpError as e:
        print(f"검색 오류: {e}")
        return []

def get_video_details(youtube, video_ids):
    videos_data = []
    for i in range(0, len(video_ids), 50):
        batch_ids = video_ids[i:i+50]
        try:
            response = youtube.videos().list(
                part='snippet,statistics,contentDetails', id=','.join(batch_ids)
            ).execute()
            
            for item in response.get('items', []):
                snippet = item.get('snippet', {})
                stats = item.get('statistics', {})
                
                try:
                    upload_date = datetime.strptime(snippet.get('publishedAt', '')[:10], '%Y-%m-%d')
                except:
                    upload_date = None
                
                videos_data.append({
                    'video_id': item.get('id'),
                    'title': snippet.get('title'),
                    'description': snippet.get('description'),
                    'url': f"https://www.youtube.com/watch?v={item.get('id')}",
                    'upload_date': upload_date.strftime('%Y-%m-%d') if upload_date else '',
                    'channel_name': snippet.get('channelTitle'),
                    'view_count': int(stats.get('viewCount', 0)),
                    'like_count': int(stats.get('likeCount', 0)),
                    'comment_count': int(stats.get('commentCount', 0)),
                    'tags': ', '.join(snippet.get('tags', [])),
                })
        except HttpError:
            pass
    return videos_data

def get_video_comments(youtube, video_id, max_comments=100):
    comments = []
    try:
        response = youtube.commentThreads().list(
            part='snippet', videoId=video_id, maxResults=min(100, max_comments), textFormat='plainText'
        ).execute()
        
        for item in response.get('items', []):
            comment = item['snippet']['topLevelComment']['snippet']
            comments.append({
                'video_id': video_id,
                'author': comment.get('authorDisplayName'),
                'text': comment.get('textDisplay'),
                'like_count': int(comment.get('likeCount', 0)),
                'published_at': comment.get('publishedAt')
            })
    except Exception:
        pass
    return comments

def crawl_youtube(youtube, keywords, start_date, end_date, max_videos, collect_comments=False, order='relevance'):
    all_videos = []
    all_comments = []
    seen_ids = set()
    
    print(f"\n검색 시작 ({len(keywords)}개 키워드)...")
    
    for keyword in keywords:
        if len(all_videos) >= max_videos:
            break
        
        print(f"검색: '{keyword}'")
        ids = search_videos(youtube, keyword, start_date, end_date, order, max_videos - len(all_videos))
        new_ids = [vid for vid in ids if vid not in seen_ids]
        seen_ids.update(new_ids)
        
        if not new_ids:
            continue
            
        videos = get_video_details(youtube, new_ids)
        
        # Filter by date and add keyword
        valid_videos = []
        for v in videos:
            if v['upload_date']:
                v_date = datetime.strptime(v['upload_date'], '%Y-%m-%d')
                if start_date <= v_date <= end_date:
                    v['search_keyword'] = keyword
                    valid_videos.append(v)
        
        all_videos.extend(valid_videos)
        print(f"  -> {len(valid_videos)}개 수집")
        
        if collect_comments:
            for v in valid_videos[:5]: # Limit comments collection
                if len(all_comments) >= max_videos * 10:
                    break
                c = get_video_comments(youtube, v['video_id'], 20)
                all_comments.extend(c)
    
    return all_videos[:max_videos], all_comments

def save_to_csv(videos, comments, output_dir='crawl_output'):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = os.path.join(output_dir + f'_{timestamp}')
    os.makedirs(output_path, exist_ok=True)
    
    df_videos = pd.DataFrame(videos)
    if not df_videos.empty:
        df_videos.to_csv(os.path.join(output_path, 'videos.csv'), index=False, encoding='utf-8-sig')
        df_videos.to_excel(os.path.join(output_path, 'videos.xlsx'), index=False)
        print(f"동영상 데이터 저장 완료: {output_path}")
    
    if comments:
        df_comments = pd.DataFrame(comments)
        df_comments.to_csv(os.path.join(output_path, 'comments.csv'), index=False, encoding='utf-8-sig')
        df_comments.to_excel(os.path.join(output_path, 'comments.xlsx'), index=False)
        print(f"댓글 데이터 저장 완료")
        
    return output_path

def main():
    try:
        keywords, start_date, end_date, max_videos, collect_comments, order = get_user_input()
        youtube = connect_youtube()
        
        videos, comments = crawl_youtube(youtube, keywords, start_date, end_date, max_videos, collect_comments, order)
        
        if not videos:
            print("수집된 데이터가 없습니다.")
            return
            
        output_dir = save_to_csv(videos, comments)
        
        # 데이터 분석 수행
        print("\n데이터 분석 시작...")
        analyze_youtube_data(os.path.join(output_dir, 'videos.csv'), os.path.join(output_dir, 'analysis_report'))
        
        if comments:
            analyze_comments_data(os.path.join(output_dir, 'comments.csv'))
            
    except Exception as e:
        print(f"\n오류 발생: {e}")
        # import traceback; traceback.print_exc() # 디버깅용

if __name__ == '__main__':
    install_requirements()
    main()

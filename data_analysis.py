"""
YouTube 크롤링 데이터 분석 및 시각화 모듈
수집된 동영상 및 댓글 데이터를 분석하고 다양한 차트를 생성합니다.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
from wordcloud import WordCloud
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
import os
from matplotlib import font_manager, rc # 폰트 세팅을 위한 모듈 추가
warnings.filterwarnings('ignore')

# 한글 폰트 설정
#plt.rcParams['font.family'] = 'DejaVu Sans'
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
            print(f"✓ {filename} 저장")
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
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>YouTube 데이터 분석 리포트</title>
    <style>
        body {{
            font-family: 'Arial', sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
            color: #333;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            text-align: center;
            color: #1f77b4;
            border-bottom: 3px solid #1f77b4;
            padding-bottom: 15px;
        }}
        h2 {{
            color: #ff7f0e;
            margin-top: 30px;
            border-left: 5px solid #ff7f0e;
            padding-left: 10px;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .stat-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .stat-card h3 {{
            margin: 0 0 10px 0;
            font-size: 14px;
            opacity: 0.9;
        }}
        .stat-card .value {{
            font-size: 24px;
            font-weight: bold;
        }}
        .chart-section {{
            margin: 40px 0;
            padding: 20px;
            background-color: #f9f9f9;
            border-radius: 8px;
        }}
        .chart-section img {{
            max-width: 100%;
            height: auto;
            border-radius: 5px;
        }}
        .link-section {{
            text-align: center;
            margin: 30px 0;
        }}
        .link-section a {{
            display: inline-block;
            padding: 12px 24px;
            margin: 10px;
            background-color: #1f77b4;
            color: white;
            text-decoration: none;
            border-radius: 5px;
            transition: background-color 0.3s;
        }}
        .link-section a:hover {{
            background-color: #ff7f0e;
        }}
        footer {{
            text-align: center;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
            color: #666;
            font-size: 12px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>YouTube 데이터 분석 리포트</h1>
        
        <h2>기본 통계</h2>
        <div class="stats-grid">
            <div class="stat-card">
                <h3>총 동영상 수</h3>
                <div class="value">{stats['총 동영상 수']}</div>
            </div>
            <div class="stat-card">
                <h3>채널 수</h3>
                <div class="value">{stats['채널 수']}</div>
            </div>
            <div class="stat-card">
                <h3>총 조회수</h3>
                <div class="value">{stats['총 조회수']}</div>
            </div>
            <div class="stat-card">
                <h3>총 좋아요</h3>
                <div class="value">{stats['총 좋아요']}</div>
            </div>
            <div class="stat-card">
                <h3>평균 조회수</h3>
                <div class="value">{stats['평균 조회수']}</div>
            </div>
            <div class="stat-card">
                <h3>평균 좋아요</h3>
                <div class="value">{stats['평균 좋아요']}</div>
            </div>
        </div>
        
        <h2>분석 결과</h2>
        
        <div class="chart-section">
            <h3>상위 10개 인기 동영상 (조회수 기준)</h3>
            <img src="01_top_videos.png" alt="상위 인기 동영상">
        </div>
        
        <div class="chart-section">
            <h3>상위 10개 채널 (동영상 개수)</h3>
            <img src="02_top_channels.png" alt="상위 채널">
        </div>
        
        <div class="chart-section">
            <h3>채널별 통계</h3>
            <img src="03_channel_statistics.png" alt="채널별 통계">
        </div>
        
        <div class="chart-section">
            <h3>날짜별 업로드 추세</h3>
            <img src="04_upload_trend.png" alt="업로드 추세">
        </div>
        
        <div class="chart-section">
            <h3>요일별 업로드 분포</h3>
            <img src="05_weekday_distribution.png" alt="요일별 분포">
        </div>
        
        <div class="chart-section">
            <h3>조회수, 좋아요, 댓글 분포</h3>
            <img src="06_views_distribution.png" alt="분포 분석">
        </div>
        
        <div class="chart-section">
            <h3>조회수 vs 좋아요 상관관계</h3>
            <img src="07_scatter_views_vs_likes.png" alt="조회수 vs 좋아요">
        </div>
        
        <div class="chart-section">
            <h3>조회수 vs 댓글 상관관계</h3>
            <img src="08_scatter_views_vs_comments.png" alt="조회수 vs 댓글">
        </div>
        
        <div class="chart-section">
            <h3>참여도 비율 분석</h3>
            <img src="09_engagement_ratio.png" alt="참여도 비율">
        </div>
        
        <div class="chart-section">
            <h3>제목 워드클라우드</h3>
            <img src="10_title_wordcloud.png" alt="제목 워드클라우드">
        </div>
        
        <div class="chart-section">
            <h3>태그 워드클라우드</h3>
            <img src="11_tags_wordcloud.png" alt="태그 워드클라우드">
        </div>
        
        <div class="link-section">
            <a href="interactive_dashboard.html" target="_blank"> 인터랙티브 대시보드 보기</a>
        </div>
        
        <footer>
            <p>이 리포트는 YouTube 크롤링 데이터 분석 도구로 생성되었습니다.</p>
            <p>생성 시간: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        </footer>
    </div>
</body>
</html>
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


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
        analyze_youtube_data(csv_file)
    else:
        print("사용법: python data_analysis.py <csv_file>")

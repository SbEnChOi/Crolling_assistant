"""
키워드 추출 모듈
문장에서 키워드를 추출하고 영어로 번역하는 기능을 제공합니다.
"""

import re
from typing import List, Tuple, Set
from konlpy.tag import Okt
from deep_translator import GoogleTranslator
import nltk
from nltk.corpus import stopwords

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
# --------------------------gpt--------------------------
# 조사 패턴 (정규표현식)
JOSA_PATTERNS = [
    r'이$', r'가$', r'을$', r'를$', r'에$', r'에서$', r'으로$', r'로$', r'의$', r'과$', r'와$',
    r'는$', r'은$', r'도$', r'만$', r'부터$', r'까지$', r'에게$', r'께$', r'한테$',
    r'에서$', r'으로부터$', r'처럼$', r'만큼$', r'보다$', r'같이$', r'마저$', r'조차$',
    r'든지$', r'이나$', r'든가$', r'라도$', r'이라도$', r'부터는$', r'부터도$'
]
# ---------------------------gpt--------------------------
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


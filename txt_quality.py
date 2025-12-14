import re
import math
from functools import lru_cache
from pymorphy3 import MorphAnalyzer

# ---------------------------------
# ЗАГРУЗКА СПИСКОВ ИМЕН / СЛОВАРЕЙ
# ---------------------------------

def load_wordlist(path, split_by_comma=False):
    try:
        with open(path, "r", encoding="utf-8") as f:
            if split_by_comma:
                text = f.read().strip()
                return set(x.strip().lower() for x in text.split(",") if x.strip())
            else:
                return set(line.strip().lower() for line in f if line.strip())
    except FileNotFoundError:
        return set()

female_names = load_wordlist(r"C:\Users\1\Desktop\python_projects\ru_names\female_names_rus.txt")
male_names = load_wordlist(r"C:\Users\1\Desktop\python_projects\ru_names\male_names_rus.txt")
male_surnames = load_wordlist(r"C:\Users\1\Desktop\python_projects\ru_names\male_surnames_rus.txt")

ALL_NAMES = female_names | male_names
ALL_SURNAMES = male_surnames

BAD_WORDS = load_wordlist(
    r"C:\Users\1\Desktop\python_projects\bad_words\very-bad_words.txt",
    split_by_comma=True
)

RUSSIAN_DICT = load_wordlist(
    r"C:\Users\1\Desktop\python_projects\exceptions\russian.txt"
)

def load_reductions(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return set(line.strip().lower() for line in f if line.strip())
    except FileNotFoundError:
        return set()
        
REDUCTIONS = load_reductions(
    r"C:\Users\1\Desktop\python_projects\exceptions\reductions.txt"
)

morph = MorphAnalyzer()

# ---------------------------------
# КЭШ ДЛЯ PYMORPHY
# ---------------------------------

@lru_cache(maxsize=50000)
def morph_parse_cached(word: str):
    return morph.parse(word)[0]

# ---------------------------------
# CAPS_ALLOWED
# ---------------------------------

def load_caps_allowed(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return set(line.strip().upper() for line in f if line.strip())
    except FileNotFoundError:
        return set()

CAPS_ALLOWED = load_caps_allowed(
    r"C:\Users\1\Desktop\python_projects\exceptions\caps_allowed.txt"
)

# ---------------------------------
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ---------------------------------

def tokenize(text):
    return re.findall(r"\b[А-Яа-яЁёA-Za-z]+\b", text)

def split_sentences(text):
    # 1. защита чисел формата X.X
    text = re.sub(r"(\d)\.(\d)", r"\1<DECIMAL>\2", text)

    # 2. защита сокращений по словарю
    # каждое сокращение может быть вида: "т.д.", "т. д.", "и т.п."
    for red in REDUCTIONS:
        safe = red.replace(".", "<DOT>")
        pattern = re.escape(red)
        text = re.sub(pattern, safe, text, flags=re.IGNORECASE)

    # 3. теперь безопасно режем по завершению предложений
    parts = re.split(r"[.!?…]+", text)

    fixed = []
    for s in parts:
        s = s.replace("<DECIMAL>", ".")
        # восстановление сокращений
        for red in REDUCTIONS:
            safe = red.replace(".", "<DOT>")
            s = s.replace(safe, red)
        cleaned = s.strip()
        if cleaned:
            fixed.append(cleaned)

    return fixed

def count_syllables_ru(word):
    vowels = "аеёиоуыэюяАЕЁИОУЫЭЮЯ"
    return max(sum(1 for ch in word if ch in vowels), 1)

def is_person_name(word):
    w = word.lower()
    if w in ALL_NAMES or w in ALL_SURNAMES:
        return True
    p = morph_parse_cached(w)
    tag_str = str(p.tag)
    return ("Name" in tag_str) or ("Surn" in tag_str) or ("Patr" in tag_str)

def is_valid_word_by_morph(word):
    """Проверка валидности через pymorphy."""
    w = word.lower()
    if is_person_name(w):
        return True
    parses = morph.parse(w)
    for p in parses:
        pos = getattr(p.tag, "POS", None)
        if pos in {"NOUN", "ADJF", "ADJS", "VERB", "INFN", "PRTF", "PRTS",
                   "GRND", "NUMR", "ADVB", "NPRO", "PREP", "CONJ", "PRCL", "INTJ"}:
            return True
    return False

def detect_caps(words):
    bad_caps = []
    for w in words:
        if len(w) < 2:
            continue
        if w.isupper():
            if w.upper() in CAPS_ALLOWED:
                continue
            if is_person_name(w):
                continue
            bad_caps.append(w)
    return bad_caps

def detect_bad_words(words):
    bad = []
    for w in words:
        if w.lower() in BAD_WORDS:
            bad.append(w)
    return bad

# ---------------------------------
# МЕТРИКИ ЧИТАЕМОСТИ
# ---------------------------------

def flesch_russian(text):
    s = split_sentences(text)
    w = tokenize(text)
    if not w or not s:
        return 0
    syll = sum(count_syllables_ru(x) for x in w)
    return 206.835 - 1.3 * (len(w)/len(s)) - 60.1 * (syll/len(w))

def fog_russian(text):
    s = split_sentences(text)
    w = tokenize(text)
    if not w or not s:
        return 0
    complex_words = sum(1 for x in w if count_syllables_ru(x) >= 4)
    return 0.4 * ((len(w)/len(s)) + 100 * (complex_words/len(w)))

def fk_russian(text):
    s = split_sentences(text)
    w = tokenize(text)
    if not w or not s:
        return 0
    syll = sum(count_syllables_ru(x) for x in w)
    return 0.5 * (len(w)/len(s)) + 8.4 * (syll/len(w)) - 15.59

def ttr(text):
    w = tokenize(text)
    return len(set(w))/len(w) if w else 0

# ---------------------------------
# РАССТОЯНИЕ ЛЕВЕНШТЕЙНА
# ---------------------------------

def levenshtein(a, b):
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i]
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            curr.append(min(prev[j] + 1, curr[j-1] + 1, prev[j-1] + cost))
        prev = curr
    return prev[-1]

# ---------------------------------
# ДЕТЕКТОР ОПЕЧАТОК
# ---------------------------------

def detect_typo_errors(word):
    """Опечаточная ошибка: расстояние 1 до любого словарного слова."""
    if word in RUSSIAN_DICT:
        return False

    if word in BAD_WORDS or is_person_name(word):
        return False

    candidates = [w for w in RUSSIAN_DICT if abs(len(w) - len(word)) <= 1]

    for cand in candidates:
        if levenshtein(word, cand) == 1:
            return True

    return False

# ---------------------------------
# ОРФОГРАФИЯ
# ---------------------------------

def detect_spelling_errors(words):
    errors = []
    for w in words:
        clean = re.sub(r"[^А-Яа-яЁё-]", "", w).lower()
        if not clean:
            continue
        if is_person_name(clean):
            continue
        if clean in BAD_WORDS:
            continue
        if clean not in RUSSIAN_DICT:
            errors.append(clean)
    return errors

# ---------------------------------
# ГРАМОТНОСТЬ
# ---------------------------------

def grammar_quality(text):

    words = tokenize(text)
    n = max(1, len(words))
    score = 100.0
    penalties = []

    # A. Орфографические ошибки
    spelling_errors = detect_spelling_errors(words)
    s_count = len(spelling_errors)

    # B. Непризнанные pymorphy формы
    unrec = []
    for w in words:
        clean = re.sub(r"[^А-Яа-яЁё]", "", w).lower()
        if not clean:
            continue
        if is_person_name(clean):
            continue
        if clean not in RUSSIAN_DICT and not is_valid_word_by_morph(clean):
            unrec.append(clean)
    u_count = len(unrec)

    # C. Опечатки (НОВАЯ ЛОГИКА)
    typos = []
    for w in words:
        clean = re.sub(r"[^А-Яа-яЁё]", "", w).lower()
        if not clean:
            continue
        if detect_typo_errors(clean):
            typos.append(clean)
    t_count = len(typos)

    # Штрафы
    spelling_ratio = s_count / n
    unrec_ratio = u_count / n
    typo_ratio = t_count / n

    MAX_SPELLING_PENALTY = 90
    MAX_UNREC_PENALTY = 25
    MAX_TYPO_PENALTY = 70

    score -= spelling_ratio * MAX_SPELLING_PENALTY
    score -= unrec_ratio * MAX_UNREC_PENALTY
    score -= typo_ratio * MAX_TYPO_PENALTY

    if s_count:
        penalties.append(f"Орфографические ошибки ({s_count}): {sorted(set(spelling_errors))}")
    if u_count:
        penalties.append(f"Неузнанные формы pymorphy ({u_count}): {sorted(set(unrec))}")
    if t_count:
        penalties.append(f"Опечатки ({t_count}): {sorted(set(typos))}")

    # Остальные штрафы
    sentences = split_sentences(text)
    missing_caps = 0
    for s in sentences:
        s_strip = s.strip()
        if s_strip and not s_strip[0].isupper():
            missing_caps += 1
            penalties.append(f"Предложение не начинается с заглавной буквы: «{s_strip[:20]}…»")

    score -= min(8, missing_caps * 2)

    if not text.strip().endswith((".", "!", "?")):
        score -= 2
        penalties.append("Текст не заканчивается знаком")

    repeats = sum(1 for i in range(1, len(words)) if words[i].lower() == words[i - 1].lower())
    if repeats:
        score -= repeats * 1.5
        penalties.append("Повтор подряд стоящих слов")

    caps = detect_caps(words)
    if len(caps) >= 3:
        score -= 5
        penalties.append(f"Много слов в CAPS: {caps[:5]}")

    score = max(1.0, min(100.0, score))

    return round(score), penalties

# ---------------------------------
# ОСНОВНАЯ ФУНКЦИЯ
# ---------------------------------

def compute_quality(res):
    # 1. Грамотность (уже готова)
    g = res["Грамотность (1–100)"]

    # 2. Лексическая чистота
    l = res["Лексическая чистота (1–100)"]

    # 3. Нормализуем читаемость
    # Индекс Флеша у русских текстов нормален примерно в пределах: 10–80
    fl = normalize_score(res["Индекс Флеша"], 0, 100)
    fg = normalize_score(res["Индекс Фога"], 4, 30)
    fk = normalize_score(res["Индекс Флеша–Кинкайда"], 0, 15)

    readability = (fl + fg + fk) / 3

    # 4. Итог: взвешенная сумма
    # Грамотность важнее → больший вес
    quality = (
        g * 0.55 +
        l * 0.15 +
        readability * 0.30
    )

    return round(quality)

def quality_category(score): 
    if score >= 95:
         return "Отличное качество текста" 
    elif score >= 80: 
        return "Хорошее качество текста" 
    elif score >= 60: 
        return "Среднее качество текста" 
    elif score >= 40:
         return "Плохое качество текста" 
    else:
        return "Очень плохое качество текста"

def education_level(fk):
    if fk < 4:
        return "Уровень начальной школы или ниже"
    elif fk < 6:
        return "Уровень 5–6 классов"
    elif fk < 8:
        return "Уровень 7–9 классов"
    elif fk < 10:
        return "Уровень старших классов"
    elif fk < 12:
        return "Уровень базового взрослого текста"
    elif fk < 14:
        return "Уровень продвинутого взрослого текста"
    elif fk < 16:
        return "Научно-популярный уровень"
    else:
        return "Научный/академический уровень"

def evaluate_russian_full(text):
    words = tokenize(text)
    grammar_score, grammar_notes = grammar_quality(text)
    bad_words = detect_bad_words(words)

    lex_clean_score = max(0, 100 - len(bad_words) * 10)

    result = {
        "Количество слов": len(words),
        "Количество предложений": len(split_sentences(text)),
        "Уникальность слов": round(ttr(text), 3),
        "Индекс Флеша": round(flesch_russian(text), 3),
        "Индекс Фога": round(fog_russian(text), 3),
        "Индекс Флеша–Кинкайда": round(fk_russian(text), 3),

        "Грамотность (1–100)": grammar_score,
        "Комментарии по грамотности": grammar_notes,

        "Лексическая чистота (1–100)": lex_clean_score,
        "Обнаруженные нецензурные слова": bad_words
    }

    # ДОБАВЛЯЕМ ИТОГОВЫЕ ПОКАЗАТЕЛИ
    base_quality = compute_quality(result)
    bad_count = len(bad_words)

    # ---------------------------
    # ШТРАФЫ ЗА МАТ
    # ---------------------------
    total_penalty = 0.0   # сумма штрафов (0.20 = 20%)

    word_count = max(1, len(words))
    mat_ratio = bad_count / word_count

    # --- 1) Штраф по доле мата ---
    if mat_ratio >= 0.30:
        total_penalty += 0.50   # 50% штраф
    elif mat_ratio >= 0.15:
        total_penalty += 0.35   # 35% штраф
    elif mat_ratio >= 0.07:
        total_penalty += 0.20   # 20% штраф
    elif bad_count > 0:
        total_penalty += 0.10   # 10% штраф при мелком мате

    # --- 2) Штраф за короткие токсичные фразы ---
    if word_count <= 8 and bad_count >= 2:
        total_penalty += 0.20   # +20% штраф сверху

    # Ограничиваем максимальный штраф — не более 80%
    total_penalty = min(total_penalty, 0.80)

    # Итоговое качество
    final_quality = round(base_quality * (1 - total_penalty))
    final_quality = max(1, min(100, final_quality))

    result["Общий коэффициент качества (1–100)"] = final_quality
    result["Категория качества"] = quality_category(final_quality)

    # ---------------------------
    # УРОВЕНЬ ОБРАЗОВАНИЯ С УЧЁТОМ МАТА
    # ---------------------------
    fk_value = result["Индекс Флеша–Кинкайда"]

    if bad_count == 0:
        # обычная логика
        edu_final = education_level(fk_value)
    else:
        # мат → только разговорные уровни
        if bad_count == 1:
            edu_final = "Бытовой разговорный уровень"
        elif bad_count == 2:
            edu_final = "Низкий бытовой уровень"
        elif bad_count == 3:
            edu_final = "Уровень эмоционального жаргона"
        else:
            edu_final = "Уровень агрессивной разговорной речи"

    result["Уровень образования текста"] = edu_final

    return result

def normalize_score(value, min_val, max_val):
    """Нормализация значения в диапазон 0–100."""
    value = max(min_val, min(max_val, value))
    return 100 * (value - min_val) / (max_val - min_val)
    
# ---------------------------------
# DEMO
# ---------------------------------
if __name__ == "__main__":
    text = input("Введите текст:\n")
    res = evaluate_russian_full(text)
    print("\nАНАЛИЗ ТЕКСТА:\n")
    for k, v in res.items():
        print(f"{k}: {v}")
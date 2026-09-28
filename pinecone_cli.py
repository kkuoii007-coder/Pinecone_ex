"""CLI для работы с Pinecone: эмбеддинги, поиск, сохранение фактов.

Использование:
    python pinecone_cli.py setup      # создать индекс, если его нет
    python pinecone_cli.py upsert     # загрузить 20+ фраз (тема: автомобили/бренды)
    python pinecone_cli.py query "<текст запроса>"
    python pinecone_cli.py save "<текст с триггером Сохрани/Запомни ...>"
    python pinecone_cli.py stats      # статистика индекса
"""

import os
import sys
import json
import time
import requests
from dotenv import load_dotenv

# Используем системное хранилище сертификатов (Windows trust store),
# чтобы сертификаты антивируса/корпоративного перехвата TLS были доверенными.
try:
    import truststore
    truststore.inject_into_ssl()
except ImportError:
    pass

from pinecone import Pinecone, ServerlessSpec

# Конфигурация из .env
load_dotenv()
OR_KEY = os.getenv("OR_KEY")
OR_MODEL = os.getenv("OR_Model", "liquid/lfm-2.5-embedding-350m:free")
PINECONE_API_KEY = os.getenv("Pinecone_API")

INDEX_NAME = "vp-autos-facts"
DIMENSION = 1024  # liquid/lfm-2.5-embedding-350m отдаёт 1024-мерные векторы
METRIC = "cosine"
NAMESPACE = "default"
EMBEDDING_URL = "https://openrouter.ai/api/v1/embeddings"

# Триггеры для сохранения фактов
SAVE_TRIGGERS = ("сохрани", "запомни", "save", "remember")


def get_pinecone():
    """Инициализация клиента Pinecone."""
    if not PINECONE_API_KEY:
        raise RuntimeError("Pinecone_API не задан в .env")
    return Pinecone(api_key=PINECONE_API_KEY)


def setup_index():
    """Создаёт serverless-индекс, если он не существует."""
    pc = get_pinecone()
    existing = [i.name for i in pc.list_indexes()]
    if INDEX_NAME in existing:
        print(f"Индекс '{INDEX_NAME}' уже существует.")
        desc = pc.describe_index(INDEX_NAME)
        print(f"  dimension={desc.dimension}, metric={desc.metric}, host={desc.host}")
        return pc.Index(INDEX_NAME)

    print(f"Создаю индекс '{INDEX_NAME}' (dim={DIMENSION}, {METRIC})...")
    pc.create_index(
        name=INDEX_NAME,
        dimension=DIMENSION,
        metric=METRIC,
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    print("Индекс создан.")
    return pc.Index(INDEX_NAME)


def embed(text: str, retries: int = 5, backoff: float = 3.0) -> list:
    """Получает эмбеддинг текста через OpenRouter /v1/embeddings.

    Делает несколько попыток с экспоненциальной задержкой, т.к. у free-моделей
    жёсткий rate limit (429).
    """
    if not OR_KEY:
        raise RuntimeError("OR_KEY не задан в .env")
    payload = {
        "model": OR_MODEL,
        "input": text,
    }
    headers = {
        "Authorization": f"Bearer {OR_KEY}",
        "Content-Type": "application/json",
    }
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            resp = requests.post(
                EMBEDDING_URL,
                headers=headers,
                json=payload,
                timeout=60,
            )
        except requests.exceptions.SSLError as e:
            # Периодический SSL-сбой от антивируса/перехвата сертификатов — ретрай
            last_err = e
            print(f"  [SSL retry {attempt}/{retries}] {e}")
            time.sleep(backoff * attempt)
            continue

        if resp.status_code == 429:
            last_err = RuntimeError(f"429 rate limit: {resp.text[:200]}")
            wait = backoff * attempt
            print(f"  [rate-limit retry {attempt}/{retries}] жду {wait:.0f}s...")
            time.sleep(wait)
            continue
        if resp.status_code != 200:
            raise RuntimeError(f"Embedding failed [{resp.status_code}]: {resp.text}")

        data = resp.json()
        vec = data["data"][0]["embedding"]
        if len(vec) != DIMENSION:
            raise RuntimeError(f"Ожидал dim={DIMENSION}, получил {len(vec)}")
        return vec

    raise RuntimeError(f"Не удалось получить эмбеддинг после {retries} попыток: {last_err}")


# 20+ фраз об автомобилях/брендах/фактах.
# Каждый кортеж: (id, text, country, body, brand, fuel, drive) — все атрибуты хранятся
# в metadata для server-side фильтрации (AND по нескольким признакам).
# Поля fuel и drive: "" = неизвестно/не указано в тексте факта.
AUTOS_FACTS = [
    ("a01", "Toyota Corolla — один из самых продаваемых автомобилей в мире, выпускается с 1966 года.", "Япония", "седан", "Toyota", "бензин", ""),
    ("a02", "Ford Mustang — культовый американский мускул-кар, представлен в 1964 году.", "США", "купе", "Ford", "бензин", ""),
    ("a03", "BMW M5 — спортивный седан с двигателем V8 и мощностью более 600 лошадиных сил.", "Германия", "седан", "BMW", "бензин", ""),
    ("a04", "Tesla Model S — электрический седан с запасом хода более 600 километров.", "США", "седан", "Tesla", "электро", ""),
    ("a05", "Porsche 911 — легендарный спортивный автомобиль с оппозитным двигателем.", "Германия", "спорткар", "Porsche", "бензин", ""),
    ("a06", "Mercedes-Benz G-Class — внедорожник с рамной конструкцией, военное происхождение.", "Германия", "внедорожник", "Mercedes-Benz", "бензин", ""),
    ("a07", "Honda Civic — компактный автомобиль, известный надёжностью и экономичностью.", "Япония", "хэтчбек", "Honda", "бензин", ""),
    ("a08", "Audi Quattro — раллийный автомобиль с полным приводом, изменивший автоспорт.", "Германия", "купе", "Audi", "бензин", "полный"),
    ("a09", "Volkswagen Beetle — легендарный «жук», выпускался более 60 лет.", "Германия", "купе", "Volkswagen", "бензин", ""),
    ("a10", "Chevrolet Corvette — американский спорткар с композитным кузовом.", "США", "спорткар", "Chevrolet", "бензин", ""),
    ("a11", "Ferrari F40 — суперкар, последний автомобиль, созданный под руководством Энцо Феррари.", "Италия", "суперкар", "Ferrari", "бензин", ""),
    ("a12", "Lamborghini Aventador — суперкар с двигателем V12 и клиновидным дизайном.", "Италия", "суперкар", "Lamborghini", "бензин", ""),
    ("a13", "Subaru WRX STI — раллийный седан с оппозитным двигателем и постоянным полным приводом.", "Япония", "седан", "Subaru", "бензин", "полный"),
    ("a14", "Nissan GT-R — японский суперкар с системой полного привода ATTESA E-TS.", "Япония", "суперкар", "Nissan", "бензин", "полный"),
    ("a15", "Mazda MX-5 Miata — самый продаваемый родстер в истории автомобилестроения.", "Япония", "родстер", "Mazda", "бензин", ""),
    ("a16", "Jeep Wrangler — внедорожник, преемник военного Willys MB.", "США", "внедорожник", "Jeep", "бензин", ""),
    ("a17", "Hyundai Sonata — среднеразмерный седан корейского производства.", "Южная Корея", "седан", "Hyundai", "бензин", ""),
    ("a18", "Volvo XC90 — шведский кроссовер, акцент на безопасности и прочности кузова.", "Швеция", "внедорожник", "Volvo", "бензин", ""),
    ("a19", "Range Rover — британский люксовый внедорожник премиум-сегмента.", "Великобритания", "внедорожник", "Land Rover", "бензин", ""),
    ("a20", "Koenigsegg Jesko — шведский гиперкар мощностью свыше 1600 лошадиных сил.", "Швеция", "гиперкар", "Koenigsegg", "бензин", ""),
    ("a21", "Bugatti Chiron — французский гиперкар с двигателем W16 и скоростью свыше 400 км/ч.", "Франция", "гиперкар", "Bugatti", "бензин", ""),
    ("a22", "Toyota Supra — японский спортивный автомобиль, символ тюнинг-культуры 90-х.", "Япония", "спорткар", "Toyota", "бензин", ""),
    ("a23", "Lada Niva — советский/российский внедорожник, первый серийный кроссовер с несущим кузовом.", "Россия", "внедорожник", "Lada", "бензин", ""),
    ("a24", "McLaren F1 — британский суперкар с центральной посадкой водителя и двигателем BMW V12.", "Великобритания", "суперкар", "McLaren", "бензин", ""),
    ("a25", "Renault Clio — французский хэтчбек, многократный победитель European Car of the Year.", "Франция", "хэтчбек", "Renault", "бензин", ""),
]

# Алиасы стран для разбора фактов пользователя. Ключи — как в metadata авто.
# Значения — все варианты упоминания (прилагательные, существительные, этнонимы) в нижнем регистре.
COUNTRY_ALIASES = {
    "Германия": ("немец", "герман", "немецк"),
    "Япония": ("япон", "японск"),
    "США": ("американ", "сша", "америк"),
    "Италия": ("итальян", "италийск", "итали"),
    "Швеция": ("швед", "шведск"),
    "Великобритания": ("британ", "английск", "британск", "англ"),
    "Франция": ("француз", "франц", "французск"),
    "Южная Корея": ("корейск", "корея", "корей"),
    "Россия": ("российск", "росси", "русск", "советск"),
}

# Алиасы типов кузова для разбора фактов пользователя.
# Ключи — каноничные значения как в metadata авто.
# Внедорожник объединяет SUV/кроссоверы/вседорожники — по ожиданиям пользователя.
BODY_ALIASES = {
    "седан": ("седан", "седанн"),
    "внедорожник": ("внедорожник", "внедорожн", "suv", "кроссовер", "кроссов", "вседорожник", "джип"),
    "спорткар": ("спорткар", "спортивн"),
    "суперкар": ("суперкар",),
    "гиперкар": ("гиперкар",),
    "хэтчбек": ("хэтчбек", "хетчбек", "хэтч"),
    "родстер": ("родстер", "родст"),
    "купе": ("купе", "купэ"),
}

# Алиасы марок для разбора фактов пользователя.
# Ключи — каноничные значения как в metadata авто.
# Значения — варианты написания/упоминания в нижнем регистре.
BRAND_ALIASES = {
    "Toyota": ("toyota", "тойот"),
    "Ford": ("ford", "форд"),
    "BMW": ("bmw"),
    "Tesla": ("tesla", "тесл"),
    "Porsche": ("porsche", "порше", "порш"),
    "Mercedes-Benz": ("mercedes", "мерседес", "мерс"),
    "Honda": ("honda", "хонд"),
    "Audi": ("audi", "ауди"),
    "Volkswagen": ("volkswagen", "vw", "фольксваген", "фольц"),
    "Chevrolet": ("chevrolet", "шевроле", "шевр"),
    "Ferrari": ("ferrari", "феррари", "ферар"),
    "Lamborghini": ("lamborghini", "ламборгини", "ламбо", "ламборг"),
    "Subaru": ("subaru", "субару"),
    "Nissan": ("nissan", "ниссан"),
    "Mazda": ("mazda", "мазда"),
    "Jeep": ("jeep", "джип"),
    "Hyundai": ("hyundai", "хендай", "хёндай", "хундай"),
    "Volvo": ("volvo", "вольво"),
    "Land Rover": ("land rover", "landrover", "ленд ровер", "лендровер", "ровер"),
    "Koenigsegg": ("koenigsegg", "кенигсегг", "кёнигсегг", "кениг"),
    "Bugatti": ("bugatti", "бугатти", "бугги"),
    "Lada": ("lada", "лада"),
    "McLaren": ("mclaren", "макларен", "маклар"),
    "Renault": ("renault", "рено"),
}

# Алиасы топлива для разбора фактов пользователя.
FUEL_ALIASES = {
    "электро": ("электро", "электромобиль", "электрокар", "батарей", "батарея"),
    "бензин": ("бензин", "бензинов", "двс", "двигатель внутреннего", " petrol", "petrol"),
}

# Алиасы типа привода для разбора фактов пользователя.
DRIVE_ALIASES = {
    "полный": ("полный привод", "полным приводом", "полного привода", "4wd", "awd", "4x4", "полноприв", "полноприводн"),
    "задний": ("задний привод", "задним приводом", "заднего привода", "rwd", "заднеприв"),
    "передний": ("передний привод", "передним приводом", "переднего привода", "fwd", "переднеприв"),
}


def cmd_upsert():
    """Загружает 20+ фраз в Pinecone с эмбеддингами и метаданными."""
    index = setup_index()
    print(f"Готовлю {len(AUTOS_FACTS)} записей для upsert...")
    vectors = []
    for vid, text, country, body, brand, fuel, drive in AUTOS_FACTS:
        vec = embed(text)
        vectors.append((vid, vec, {
            "text": text,
            "country": country,
            "body": body,
            "brand": brand,
            "fuel": fuel,
            "drive": drive,
            "type": "car",
        }))
        print(f"  [embedded] {vid}: {text[:50]}...")
        time.sleep(1.0)  # бережём free rate-limit между запросами

    # upsert батчами по 50 (меньше чем всего записей — одним батчем)
    index.upsert(vectors=vectors, namespace=NAMESPACE)
    print(f"Загружено {len(vectors)} векторов в namespace='{NAMESPACE}'.")


def _fetch_user_facts(index, query_vec, top_k=10):
    """Достаёт сохранённые факты пользователя (type=fact), семантически близкие к запросу.

    Факты хранятся с metadata {"type": "fact"}, авто — без поля type.
    Возвращает список текстов фактов.
    """
    res = index.query(
        vector=query_vec,
        top_k=top_k,
        include_metadata=True,
        namespace=NAMESPACE,
        filter={"type": {"$eq": "fact"}},
    )
    facts = []
    for m in res.matches:
        mtext = m.metadata.get("text", "") if m.metadata else ""
        if mtext:
            facts.append(mtext)
    return facts


def _extract_filters(facts):
    """Разбирает тексты фактов и возвращает словарь фильтров.

    Ищет упоминания по всем признакам: country, body, brand, fuel, drive.
    Берёт первое совпадение по каждому признаку. Значения — каноничные для metadata.
    Пустой/неизвестный признак в результат не попадает.
    """
    result = {}
    if not facts:
        return result
    joined = " " + " ".join(facts).lower() + " "
    # Порядок важен: более специфичные (brand) первыми не мешают общим (country, body).
    alias_maps = [
        ("country", COUNTRY_ALIASES),
        ("body", BODY_ALIASES),
        ("brand", BRAND_ALIASES),
        ("fuel", FUEL_ALIASES),
        ("drive", DRIVE_ALIASES),
    ]
    for key, alias_map in alias_maps:
        for canon, aliases in alias_map.items():
            for alias in aliases:
                if alias in joined:
                    result[key] = canon
                    break
            if key in result:
                break
    return result


def cmd_query(text: str, top_k: int = 5):
    """Семантический поиск по смыслу с учётом сохранённых фактов пользователя.

    Этап 1: достаём факты пользователя, близкие к исходному запросу.
    Этап 2: обогащаем запрос текстами фактов и разбираем фильтры
            (country, body, brand, fuel, drive).
    Этап 3: ищем авто по обогащённому запросу.
            Все извлечённые фильтры собираются в один Pinecone filter (AND),
            применяется на стороне сервера. Сами факты (type=fact) всегда исключаются.
    """
    index = setup_index()
    vec = embed(text)

    # Этап 1: факты пользователя
    facts = _fetch_user_facts(index, vec)

    # Этап 2: обогащение запроса + разбор фильтров
    if facts:
        enriched = text + " " + " ".join(facts)
        print("\nУчтены факты пользователя:")
        for f in facts:
            print(f"  - {f}")
        query_vec = embed(enriched)
        filters = _extract_filters(facts)
    else:
        query_vec = vec
        filters = {}

    # Этап 3: формирование AND-фильтра и поиск авто
    # type != fact — всегда исключаем сами факты из выдачи.
    # Все извлечённые признаки ограничивают выдачу (AND на стороне Pinecone).
    base_filter = {"type": {"$ne": "fact"}}
    if filters:
        for key in ("country", "body", "brand", "fuel", "drive"):
            val = filters.get(key)
            if val:
                base_filter[key] = {"$eq": val}
                print(f"  -> жёсткий фильтр по {key}: {val}")

    fetch = max(top_k * 3, top_k + 10)
    res = index.query(
        vector=query_vec,
        top_k=fetch,
        include_metadata=True,
        namespace=NAMESPACE,
        filter=base_filter,
    )
    autos = res.matches[:top_k]

    print(f"\nЗапрос: '{text}'")
    print(f"Найдено совпадений: {len(autos)}")
    for i, m in enumerate(autos, 1):
        score = m.score
        mid = m.id
        md = m.metadata or {}
        print(f"  {i}. id={mid} | score={score:.4f} | "
              f"country={md.get('country','')} | body={md.get('body','')} | "
              f"brand={md.get('brand','')} | fuel={md.get('fuel','')} | "
              f"drive={md.get('drive','')}")
        print(f"     text: {md.get('text','')}")


def cmd_save(text: str):
    """Сохраняет факт, если текст содержит триггер 'Сохрани'/'Запомни'.

    Пример: python pinecone_cli.py save "Сохрани: пользователь любит спорткары"
    """
    lower = text.lower()
    triggered = any(t in lower for t in SAVE_TRIGGERS)
    if not triggered:
        print("Триггер 'Сохрани'/'Запомни' не найден. Факт не сохранён.")
        print("Начните сообщение с одного из слов:", ", ".join(SAVE_TRIGGERS))
        return

    # Очищаем текст от триггера
    clean = text
    for t in SAVE_TRIGGERS:
        if clean.lower().startswith(t):
            # убираем триггер и разделитель
            rest = clean[len(t):]
            clean = rest.lstrip(": =-—").strip()
            break
    if not clean:
        print("После триггера нет текста для сохранения.")
        return

    index = setup_index()
    # Генерируем id на основе счётчика в индексе
    stats = index.describe_index_stats()
    total = stats.total_vector_count if stats.total_vector_count else 0
    vid = f"fact-{total + 1:04d}"
    vec = embed(clean)
    index.upsert(vectors=[(vid, vec, {"text": clean, "type": "fact"})], namespace=NAMESPACE)
    print(f"Факт сохранён: id={vid}")
    print(f"  text: {clean}")


def cmd_stats():
    """Выводит статистику индекса."""
    index = setup_index()
    stats = index.describe_index_stats()
    print(f"Индекс: {INDEX_NAME}")
    print(f"  total_vector_count: {stats.total_vector_count}")
    if stats.namespaces:
        for ns, ns_stat in stats.namespaces.items():
            print(f"  namespace '{ns}': vector_count={ns_stat.vector_count}")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    cmd = sys.argv[1].lower()
    if cmd == "setup":
        setup_index()
    elif cmd == "upsert":
        cmd_upsert()
    elif cmd == "query":
        if len(sys.argv) < 3:
            print("Использование: python pinecone_cli.py query \"<текст>\"")
            sys.exit(1)
        cmd_query(sys.argv[2])
    elif cmd == "save":
        if len(sys.argv) < 3:
            print("Использование: python pinecone_cli.py save \"<текст с триггером>\"")
            sys.exit(1)
        cmd_save(sys.argv[2])
    elif cmd == "stats":
        cmd_stats()
    elif cmd == "test-embed":
        if len(sys.argv) < 3:
            print("Использование: python pinecone_cli.py test-embed \"<текст>\"")
            sys.exit(1)
        v = embed(sys.argv[2])
        print(f"dim={len(v)}, первые 5 значений: {v[:5]}")
    else:
        print(f"Неизвестная команда: {cmd}")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()

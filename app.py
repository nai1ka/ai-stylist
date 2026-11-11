import hashlib
import logging

import streamlit as st

from stylist.body import SHAPES, classify_body
from stylist.face import ModelUnavailableError, analyze_face
from stylist.images import InvalidImageError, load_image
from stylist.recommendations import get_clothing, get_hairstyles
from stylist.schemas import BodyMeasurements
from stylist.colors import build_color_regions, render_masks, render_region_mask
from stylist.segmentation import load_face_parser, segment_face
from stylist.seasonal import load_seasonal_model, analyze_seasonal, SEASON_LABELS
from stylist.seasonal_preprocessing import prepare_seasonal_image

logging.basicConfig(filename="stylist.log", level=logging.ERROR)
st.set_page_config(page_title="AI-Stylist", page_icon="👗", layout="centered")
FACE_LABELS = {"heart": "Сердцевидная", "oblong": "Продолговатая", "oval": "Овальная", "round": "Круглая", "square": "Квадратная"}
BODY_IDS = list(SHAPES)
STYLE_LABELS = {"Деловой": "business", "Повседневный": "casual", "Спортивный": "sporty", "Неформальный": "informal"}
LENGTHS = {"Без ограничения": "any", "Короткая": "short", "Средняя": "medium", "Длинная": "long"}

st.title("AI-Stylist")


@st.cache_resource(show_spinner="Загружаю локальную модель сегментации…")
def cached_face_parser():
    return load_face_parser()

@st.cache_resource(show_spinner="Загружаю локальную модель цветотипа…")
def cached_seasonal_model():
    return load_seasonal_model()


def reset_photo_results():
    for key in ("face_analysis", "face_crop", "face_manual", "face_choice", "color_analysis",
                "seasonal_analysis", "seasonal_input", "seasonal_manual", "seasonal_choice"):
        st.session_state.pop(key, None)


upload = st.file_uploader("Фото лица (JPEG или PNG, до 10 МБ)", type=["jpg", "jpeg", "png"])
if upload:
    photo_bytes = upload.getvalue()
    digest = hashlib.sha256(photo_bytes).hexdigest()
    if st.session_state.get("photo_digest") != digest:
        st.session_state.photo_digest = digest
        reset_photo_results()
    try:
        image = load_image(photo_bytes)
        st.image(image, caption="Загруженное фото", use_container_width=True)
    except InvalidImageError as error:
        st.error(str(error))
        image = None
    if image and st.button("Начать анализ", type="primary"):
        reset_photo_results()
        try:
            with st.spinner("Шаг 1 из 3: анализирую лицо…"):
                face_analysis, face_crop = analyze_face(image)
                st.session_state.face_analysis = face_analysis
                st.session_state.face_crop = face_crop
                st.session_state.pop("face_manual", None)
            with st.spinner("Шаг 2 из 3: выделяю кожу, волосы и области глаз на CPU…"):
                parser, parser_model = cached_face_parser()
                parsing = segment_face(image, parser, parser_model)
                regions = build_color_regions(image, parsing)
                st.session_state.color_analysis = (digest, parsing, regions)
            try:
                with st.spinner("Шаг 3 из 3: определяю сезонный цветотип на CPU…"):
                    prepared = prepare_seasonal_image(image, parsing)
                    st.session_state.seasonal_input = prepared
                    seasonal_model, revision = cached_seasonal_model()
                    st.session_state.seasonal_analysis = analyze_seasonal(prepared, seasonal_model, revision)
            except Exception as error:
                st.error(str(error))
                logging.exception("Seasonal analysis failed")
            st.success("Анализ фото завершён.")
        except Exception as error:
            st.error(str(error))
            logging.exception("Photo analysis failed")
    st.markdown("#### Результаты анализа")
    color_result = st.session_state.get("color_analysis")
    if color_result and color_result[0] == digest:
        _, parsing, regions = color_result
        with st.expander("Разметка лица"):
            st.image(render_masks(image, regions), caption="Кожа, волосы и глаза", use_container_width=True)
            columns = st.columns(len(regions))
            for column, region in zip(columns, regions):
                with column:
                    st.image(render_region_mask(image, region), caption=region.label, use_container_width=True)
                    if region.median_rgb is None:
                        st.warning("Область не найдена")
                    else:
                        rgb = region.median_rgb
                        st.markdown(
                            f'<div style="height:2.2rem;border-radius:.5rem;background:rgb{rgb};border:1px solid #777"></div>',
                            unsafe_allow_html=True,
                        )
                        st.caption(f"Медиана RGB: {rgb[0]}, {rgb[1]}, {rgb[2]}")
            st.caption("Центр зрачка отмечен приблизительно: модель выделяет область глаза целиком.")
    prepared = st.session_state.get("seasonal_input")
    if prepared is not None:
        with st.expander("Изображение для определения цветотипа"):
            st.image(prepared, caption="Фактический вход 224×224 до нормализации")
            st.caption("Маска и обрезка — предварительная аппроксимация обработки автора. Рот и губы дополнительно сохраняются по контуру MediaPipe.")
    seasonal = st.session_state.get("seasonal_analysis")
    if seasonal:
        st.write(f"Предполагаемый цветотип: **{SEASON_LABELS[seasonal.predicted_season]}**")
        st.write("Оценки модели: " + "; ".join(f"{SEASON_LABELS[k]} — {v:.1%}" for k, v in seasonal.scores.items()))
        st.caption("Результат зависит от освещения и обработки фотографии. Категория условная, её можно исправить вручную.")
        choices = ["Использовать результат модели", *SEASON_LABELS.values()]
        choice = st.selectbox("Цветотип для стиль-карты", choices, key="seasonal_choice")
        st.session_state.seasonal_manual = next((k for k, v in SEASON_LABELS.items() if v == choice), None)
    face = st.session_state.get("face_analysis")
    if face:
        st.subheader("Форма лица")
        st.write(f"Модель предполагает: **{FACE_LABELS[face.predicted_shape]}**")
        st.bar_chart({FACE_LABELS[key]: value for key, value in face.scores.items()})
        with st.expander("Точки MediaPipe"):
            st.image(st.session_state.face_crop)
        face_options = ["Использовать результат модели", *FACE_LABELS.values()]
        manual = st.selectbox("Форма лица для рекомендаций", face_options, key="face_choice")
        st.session_state.face_manual = None if manual == face_options[0] else next(k for k, v in FACE_LABELS.items() if v == manual)

else:
    reset_photo_results()
    st.session_state.pop("photo_digest", None)

st.header("Мерки и предпочтения")
with st.form("measurements"):
    first, second, third = st.columns(3)
    bust = first.number_input("Грудь, см", min_value=1.0, value=90.0, step=1.0)
    waist = second.number_input("Талия, см", min_value=1.0, value=70.0, step=1.0)
    hips = third.number_input("Бёдра, см", min_value=1.0, value=96.0, step=1.0)
    st.caption("Мерки в сантиметрах.")
    style_label = st.selectbox("Стиль одежды", list(STYLE_LABELS))
    length_label = st.selectbox("Длина причёски", list(LENGTHS))
    submitted = st.form_submit_button("Рассчитать тип фигуры")

if submitted:
    st.session_state.body_analysis = classify_body(BodyMeasurements(bust, waist, hips))
    st.session_state.body_manual = None
    st.session_state.preferences = (STYLE_LABELS[style_label], LENGTHS[length_label])

body = st.session_state.get("body_analysis")
if body:
    st.write(f"Расчёт: **{SHAPES[body.predicted_shape] if body.predicted_shape else 'однозначная категория не определена'}**")
    if body.explanation:
        st.caption(body.explanation)
    options = ["Использовать результат расчёта", *SHAPES.values()]
    selected = st.selectbox("Тип фигуры для рекомендаций", options, key="body_choice")
    st.session_state.body_manual = None if selected == options[0] else BODY_IDS[list(SHAPES.values()).index(selected)]

if st.button("Сформировать стиль-карту"):
    face = st.session_state.get("face_analysis")
    face_shape = st.session_state.get("face_manual") or (face.predicted_shape if face else None)
    body = st.session_state.get("body_analysis")
    body_shape = st.session_state.get("body_manual") or (body.predicted_shape if body else None)
    preferences = st.session_state.get("preferences")
    if not face_shape or not body_shape or not preferences:
        st.warning("Сначала выполните анализ фото и расчёт мерок либо выберите категории вручную.")
    else:
        style, length = preferences
        st.subheader("Ваша стиль-карта")
        seasonal = st.session_state.get("seasonal_analysis")
        season = st.session_state.get("seasonal_manual") or (seasonal.predicted_season if seasonal else None)
        if season:
            source = "вручную" if st.session_state.get("seasonal_manual") else "по модели"
            st.write(f"Цветотип: {SEASON_LABELS[season]} ({source}).")
        st.write(f"Лицо: {FACE_LABELS[face_shape]} ({'вручную' if st.session_state.get('face_manual') else 'по модели'}).")
        st.write(f"Фигура: {SHAPES[body_shape]} ({'вручную' if st.session_state.get('body_manual') else 'по меркам'}).")
        st.write(f"Предпочтения: {next(k for k,v in STYLE_LABELS.items() if v == style)} стиль, длина волос — {next(k for k,v in LENGTHS.items() if v == length)}.")
        st.markdown("#### Причёски")
        styles = get_hairstyles(face_shape, length)
        if not styles:
            st.info("В базе пока нет подтверждённых рекомендаций для такого сочетания.")
        for item in styles:
            st.markdown(f"**{item.title}.** {item.description} {item.reason}")
            st.caption("Источник: " + ", ".join(item.source_ids) + " — подробности в SOURCES.md")
        st.markdown("#### Одежда")
        clothes = get_clothing(body_shape, style)
        if not clothes:
            st.info("В базе пока нет подтверждённых рекомендаций для такого сочетания.")
        for item in clothes:
            st.markdown(f"**{item.title}.** {item.description} {item.reason}")
            st.caption("Источник: " + ", ".join(item.source_ids) + " — подробности в SOURCES.md")

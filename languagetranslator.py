import gradio as gr
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

# Load the model and tokenizer
model = MBartForConditionalGeneration.from_pretrained("SnypzZz/Llama2-13b-Language-translate")
tokenizer = MBart50TokenizerFast.from_pretrained("SnypzZz/Llama2-13b-Language-translate", src_lang="en_XX")

# Language options for dropdown
languages = {
    'Arabic': 'ar_AR', 'Czech': 'cs_CZ', 'German': 'de_DE', 'English': 'en_XX', 'Spanish': 'es_XX', 
    'Estonian': 'et_EE', 'Finnish': 'fi_FI', 'French': 'fr_XX', 'Gujarati': 'gu_IN', 'Hindi': 'hi_IN',
    'Italian': 'it_IT', 'Japanese': 'ja_XX', 'Kazakh': 'kk_KZ', 'Korean': 'ko_KR', 'Lithuanian': 'lt_LT',
    'Latvian': 'lv_LV', 'Burmese': 'my_MM', 'Nepali': 'ne_NP', 'Dutch': 'nl_XX', 'Romanian': 'ro_RO',
    'Russian': 'ru_RU', 'Sinhala': 'si_LK', 'Turkish': 'tr_TR', 'Vietnamese': 'vi_VN', 'Chinese': 'zh_CN',
    'Afrikaans': 'af_ZA', 'Azerbaijani': 'az_AZ', 'Bengali': 'bn_IN', 'Persian': 'fa_IR', 'Hebrew': 'he_IL',
    'Croatian': 'hr_HR', 'Indonesian': 'id_ID', 'Georgian': 'ka_GE', 'Khmer': 'km_KH', 'Macedonian': 'mk_MK',
    'Malayalam': 'ml_IN', 'Mongolian': 'mn_MN', 'Marathi': 'mr_IN', 'Polish': 'pl_PL', 'Pashto': 'ps_AF',
    'Portuguese': 'pt_XX', 'Swedish': 'sv_SE', 'Swahili': 'sw_KE', 'Tamil': 'ta_IN', 'Telugu': 'te_IN',
    'Thai': 'th_TH', 'Tagalog': 'tl_XX', 'Ukrainian': 'uk_UA', 'Urdu': 'ur_PK', 'Xhosa': 'xh_ZA',
    'Galician': 'gl_ES', 'Slovene': 'sl_SI'
}

# Translation function
def translate_text(text, target_lang):
    model_inputs = tokenizer(text, return_tensors="pt")
    generated_tokens = model.generate(
        **model_inputs, 
        forced_bos_token_id=tokenizer.lang_code_to_id[languages[target_lang]]
    )
    translated_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    return translated_text[0]

# Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("## Multilingual Translation App")
    with gr.Row():

        with gr.Column():
            language_dropdown = gr.Dropdown(choices=list(languages.keys()), label="Select target language")

            text_input = gr.Textbox(label="Enter text to translate", placeholder="Type text here...", lines=4)
    
            translate_button = gr.Button("Translate")
        output_text = gr.Textbox(label="Translated text", lines=4)
    print(output_text)
    translate_button.click(fn=translate_text, inputs=[text_input, language_dropdown], outputs=output_text)

demo.launch()
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   
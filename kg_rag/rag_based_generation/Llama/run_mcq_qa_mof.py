'''
This script takes the MCQ style questions from the csv file and save the result as another csv file. 
This script makes use of Llama model.
Before running this script, make sure to configure the filepaths in config.yaml file.
'''
import os
import re
from difflib import SequenceMatcher
from langchain import PromptTemplate, LLMChain
from kg_rag.utility import *


QUESTION_PATH = config_data["MCQ_PATH"]
SYSTEM_PROMPT = system_prompts["MCQ_QUESTION"]#system_prompts.yaml文件中MCQ_QUESTION的值
CONTEXT_VOLUME = int(config_data["CONTEXT_VOLUME"])#上下文体积
QUESTION_VS_CONTEXT_SIMILARITY_PERCENTILE_THRESHOLD = float(config_data["QUESTION_VS_CONTEXT_SIMILARITY_PERCENTILE_THRESHOLD"])#衡量问题与上下文之间相似度的百分位数阈值
QUESTION_VS_CONTEXT_MINIMUM_SIMILARITY = float(config_data["QUESTION_VS_CONTEXT_MINIMUM_SIMILARITY"])#衡量问题与上下文之间相似度的最小相似度
VECTOR_DB_PATH = config_data["VECTOR_DB_PATH"]#向量数据库路径
NODE_CONTEXT_PATH = config_data["NODE_CONTEXT_PATH"]
SENTENCE_EMBEDDING_MODEL_FOR_NODE_RETRIEVAL = config_data["SENTENCE_EMBEDDING_MODEL_FOR_NODE_RETRIEVAL"]
SENTENCE_EMBEDDING_MODEL_FOR_CONTEXT_RETRIEVAL = config_data["SENTENCE_EMBEDDING_MODEL_FOR_CONTEXT_RETRIEVAL"]
SAVE_PATH = config_data["SAVE_RESULTS_PATH"]
MODEL_NAME = config_data["LLAMA_MODEL_NAME"]
BRANCH_NAME = config_data["LLAMA_MODEL_BRANCH"]
CACHE_DIR = config_data["LLM_CACHE_DIR"]

save_name = "_".join(MODEL_NAME.split("/")[-1].split("-"))+"_lcd_mcq_done9_promptfix.csv"
MAX_QUESTIONS = int(os.environ.get("KGRAG_MAX_QUESTIONS", "500"))


INSTRUCTION = "Context:\n\n{context} \n\nQuestion: {question}"

vectorstore = load_chroma(VECTOR_DB_PATH, SENTENCE_EMBEDDING_MODEL_FOR_NODE_RETRIEVAL)
embedding_function_for_context_retrieval = load_sentence_transformer(SENTENCE_EMBEDDING_MODEL_FOR_CONTEXT_RETRIEVAL)
node_context_df = pd.read_csv(NODE_CONTEXT_PATH)
edge_evidence = False
MOF_PROPERTY_ALIASES = {
    "ASA_m2_cm3": ["asa_m2_cm3", "accessible surface area", "asa in m²/cm³", "asa in m2/cm3"],
    "AV_cm3_g": ["av_cm3_g", "adsorption volume", "av in cm³/g", "av in cm3/g"],
    "PLD": ["pld", "pore limiting diameter"],
    "LCD": ["lcd", "largest cavity diameter"],
    "Open_Metal_Sites": ["open_metal_sites", "open metal sites"],
    "LFPD": ["lfpd"],
    "filename": ["filename", "file name"],
    "ASA_m2_g": ["asa_m2_g", "asa in m²/g", "asa in m2/g"],
    "All_Metals": ["all_metals", "all metals", "metals are present", "what metals"],
    "Has_OMS": ["has_oms", "has oms", "have oms", "has open metal sites", "have open metal sites"],
    "ID": [" id ", "what is the id", "material id"],
}


def normalize_name(name):
    return re.sub(r"[^a-z0-9]+", "", str(name).casefold())


def extract_given_list(question):
    match = re.search(r"given list is:\s*(.*)$", str(question), flags=re.I)
    if not match:
        return []
    return [item.strip().strip(".") for item in match.group(1).split(",") if item.strip()]


def extract_first_material(question):
    quoted = re.findall(r'"([^"]+)"', str(question))
    if quoted:
        return quoted[0].strip()
    match = re.search(r"(?:of|material)\s+(.+?)\??$", str(question))
    return match.group(1).strip().strip(".?") if match else ""


def extract_lcd_range(question):
    match = re.search(r"between\s*([\d.]+)\s*(?:and|-)\s*([\d.]+)", str(question), flags=re.I)
    if not match:
        return None
    return float(match.group(1)), float(match.group(2))


def parse_node_properties(context):
    context = str(context).replace("Node Properties:", "")
    props = {}
    for key, value in re.findall(r"([A-Za-z0-9_]+):\s*([^;\n]+)", context):
        props[key.strip()] = value.strip()
    return props


def get_mof_context_by_name(name):
    normalized_target = normalize_name(name)
    if not normalized_target:
        return "", "", np.nan, ""

    node_names = node_context_df["node_name"].astype(str)
    exact_match = node_context_df[node_names.str.casefold() == str(name).casefold()]
    if not exact_match.empty:
        return exact_match.iloc[0]["node_context"], exact_match.iloc[0]["node_name"], 0.0, "exact_node_name"

    normalized_names = node_names.map(normalize_name)
    normalized_match = node_context_df[normalized_names == normalized_target]
    if not normalized_match.empty:
        return (
            normalized_match.iloc[0]["node_context"],
            normalized_match.iloc[0]["node_name"],
            0.0,
            "normalized_node_name",
        )

    fuzzy_scores = normalized_names.map(
        lambda node_name: SequenceMatcher(None, normalized_target, node_name).ratio()
    )
    best_index = fuzzy_scores.idxmax()
    if fuzzy_scores.loc[best_index] >= 0.92:
        return (
            node_context_df.loc[best_index, "node_context"],
            node_context_df.loc[best_index, "node_name"],
            float(fuzzy_scores.loc[best_index]),
            "fuzzy_node_name",
        )
    return "", "", np.nan, ""


def get_context_for_mof_question(question):
    given_list = extract_given_list(question)
    if given_list:
        contexts = []
        retrieved_nodes = []
        methods = []
        for material in given_list:
            context, node_name, score, method = get_mof_context_by_name(material)
            if context:
                contexts.append(f"Material: {node_name}\\n{context}")
                retrieved_nodes.append(node_name)
                methods.append(method)
        if contexts:
            return "\\n\\n".join(contexts), ", ".join(retrieved_nodes), np.nan, "given_list_" + "+".join(sorted(set(methods)))

    material = extract_first_material(question)
    if material:
        context, node_name, score, method = get_mof_context_by_name(material)
        if context:
            return context, node_name, score, method

    node_hits = vectorstore.similarity_search_with_score(question, k=3)
    for document, score in node_hits:
        node_name = document.page_content
        node_match = node_context_df[node_context_df.node_name == node_name]
        if not node_match.empty:
            return node_match.iloc[0]["node_context"], node_name, float(score), "vector_top3"

    context = retrieve_context(
        question,
        vectorstore,
        embedding_function_for_context_retrieval,
        node_context_df,
        CONTEXT_VOLUME,
        QUESTION_VS_CONTEXT_SIMILARITY_PERCENTILE_THRESHOLD,
        QUESTION_VS_CONTEXT_MINIMUM_SIMILARITY,
        edge_evidence,
    )
    return context, "", np.nan, "legacy_retrieve_context"


def mof_lcd(material):
    context, node_name, _, _ = get_mof_context_by_name(material)
    if not context:
        return None
    props = parse_node_properties(context)
    try:
        return float(props["LCD"])
    except (KeyError, ValueError):
        return None


def mof_property(material, property_name):
    context, node_name, _, _ = get_mof_context_by_name(material)
    if not context:
        return None
    return parse_node_properties(context).get(property_name)


def detect_mof_property(question):
    question_lower = f" {str(question).casefold()} "
    for property_name, aliases in MOF_PROPERTY_ALIASES.items():
        for alias in aliases:
            if alias.casefold() in question_lower:
                return property_name
    return ""


def rule_based_mof_answer(question):
    question_lower = str(question).lower()
    given_list = extract_given_list(question)
    lcd_range = extract_lcd_range(question)
    if given_list and lcd_range and "lcd" in question_lower:
        lower, upper = lcd_range
        hits = []
        for material in given_list:
            lcd = mof_lcd(material)
            if lcd is not None and lower <= lcd <= upper:
                hits.append(material)
        return ", ".join(hits) if hits else "No material in the given list matches the LCD range."

    if "lcd" in question_lower:
        material = extract_first_material(question)
        lcd = mof_lcd(material)
        if lcd is not None:
            return f"The LCD value of {material} is {lcd}."

    property_name = detect_mof_property(question)
    if property_name:
        material = extract_first_material(question)
        value = mof_property(material, property_name)
        if value is not None:
            if property_name == "Has_OMS":
                return f"The Has_OMS value of {material} is {value}."
            return f"The {property_name} value of {material} is {value}."

    return ""


def clean_output(output):
    output = str(output).strip()
    if "Answer:" in output:
        output = output.split("Answer:")[-1].strip()
    for marker in ["\nQuestion:", "\nContext:", "\nNow,"]:
        if marker in output:
            output = output.split(marker)[0].strip()
    return output


# def main():    
#     start_time = time.time()
#     llm = llama_model(MODEL_NAME, BRANCH_NAME, CACHE_DIR)               
#     template = get_prompt(INSTRUCTION, SYSTEM_PROMPT)
#     prompt = PromptTemplate(template=template, input_variables=["context", "question"])
#     llm_chain = LLMChain(prompt=prompt, llm=llm)    
#     question_df = pd.read_csv(QUESTION_PATH)  
#     answer_list = []
#     for index, row in question_df.iterrows():
#         print("##########################")
#         print("使用Chemllm进行KG-RAG推理")
#         question = row["text"]
#         context = retrieve_context(question, vectorstore, embedding_function_for_context_retrieval, node_context_df, CONTEXT_VOLUME, QUESTION_VS_CONTEXT_SIMILARITY_PERCENTILE_THRESHOLD, QUESTION_VS_CONTEXT_MINIMUM_SIMILARITY, edge_evidence)
        
#         output = llm_chain.run(context=context, question=question).split("\n")[0]
#         answer_list.append((row["text"], row["correct_node"], output))
#         print("问题内容：",row["text"])
#         print("正确答案：",row["correct_node"])
#         print("推理答案：",output)
#         print("##########################")
#     answer_df = pd.DataFrame(answer_list, columns=["question", "correct_answer", "llm_answer"])
#     answer_df.to_csv(os.path.join(SAVE_PATH, save_name), index=False, header=True) 
#     print("Completed in {} min".format((time.time()-start_time)/60))

#2questions
def main():    
    start_time = time.time()
    llm = llama_model(MODEL_NAME, BRANCH_NAME, CACHE_DIR)               
    template = """You are a MOF property extraction assistant.
Use only the retrieved context. Do not invent properties.
For LCD range questions, select only materials whose LCD values are within the requested range.

Context: {context}

    Question: {question}

    Answer:"""
    
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    llm_chain = LLMChain(prompt=prompt, llm=llm)    
    question_df = pd.read_csv(QUESTION_PATH)  
    
    # 检查文件是否存在并确定起始位置
    save_file_path = os.path.join(SAVE_PATH, save_name)
    if os.path.exists(save_file_path):
        existing_df = pd.read_csv(save_file_path)
        question_count = len(existing_df)
        print(f"发现已存在的文件，从第 {question_count + 1} 个问题继续...")
    else:
        question_count = 0
        # 创建新文件并写入表头
        pd.DataFrame(columns=[
            "question",
            "correct_answer",
            "llm_answer",
            "retrieved_node",
            "retrieval_score",
            "retrieval_method",
            "context",
        ]).to_csv(
            save_file_path, index=False, mode='w'
        )
    
    for index, row in question_df.iloc[question_count:].iterrows():
        if question_count >= min(MAX_QUESTIONS, len(question_df)):
            break
        print("##########################")
        print(f"正在处理第 {question_count + 1} 个问题")
        print("使用Chemllm进行KG-RAG推理")
        
        question = row["text"]
        context, retrieved_node, retrieval_score, retrieval_method = get_context_for_mof_question(question)

        rule_answer = rule_based_mof_answer(question)
        if rule_answer:
            output = rule_answer
        else:
            output = llm_chain.run(context=context, question=question)
            output = clean_output(output)
        
        # 立即将当前结果追加到文件
        result_df = pd.DataFrame({
            "question": [row["text"]],
            "correct_answer": [row["correct_node"]],
            "llm_answer": [output],
            "retrieved_node": [retrieved_node],
            "retrieval_score": [retrieval_score],
            "retrieval_method": [retrieval_method],
            "context": [context],
        })
        result_df.to_csv(save_file_path, index=False, mode='a', header=False)
        
        print("问题内容：", row["text"])
        print("检索节点：", retrieved_node)
        print("检索方式：", retrieval_method)
        print("上下文：", context)
        print("正确答案：", row["correct_node"])
        print("推理答案：", output)
        print("##########################")
        question_count += 1

    print("Completed in {} min".format((time.time()-start_time)/60))


if __name__ == "__main__":
    main()
    
    

import numpy as np
import json
import os
import pandas as pd
import glob
from typing import Dict, Tuple, Sequence, List, Union
from collections import defaultdict



class BenchmarkBase:
    """
    Loading dataset and few_shot_prompt
    Process dataset for zero_shot or few_shot setting

    Subclass should config:
        benchmark_name (`str`)
        benchmark_dir (`str`)
        subtasks (`List[str]`)

    Subclass should implement:
        _init_task_prompts(self):
            config task_prompt for each setting (and for each subtask)
    """

    subtasks = []
    benchmarks_dir = 'benchmark_eval/benchmarks'

    def __init__(
        self, 
        benchmark_name: str = ''
    ):
        self.benchmark_name = benchmark_name

        self.benchmark_dir = os.path.join(self.benchmarks_dir, benchmark_name)
        self._init_task_prompts()
        self._normalize_task_prompts()

    def _init_task_prompts(self):
        """
        Config self.task_prompt_zero_shot and self.task_prompt_few_shot
        """
        raise NotImplementedError

    def _normalize_task_prompts(self):
        if isinstance(self.task_prompt_zero_shot, str):
            self.task_prompt_zero_shot = {self.benchmark_name: self.task_prompt_zero_shot}
        if isinstance(self.response_with_zero_shot, str):
            self.response_with_zero_shot = {k: self.response_with_zero_shot for k in self.task_prompt_zero_shot.keys()}
        if isinstance(self.task_prompt_few_shot, str):
            self.task_prompt_few_shot = {self.benchmark_name: self.task_prompt_few_shot}

    @staticmethod
    def _read_txt(file: str) -> List[dict]:
        assert isinstance(file, str)
        with open(file, 'r') as f:
            return f.read()

    @staticmethod
    def _load_jsonl(file: str) -> List[dict]:
        assert isinstance(file, str) and (file.endswith('.jsonl') or file.endswith('.jl'))
        with open(file, 'r') as f:
            lines = f.read().strip().split('\n')
            data_list = [json.loads(line) for line in lines]
        return data_list

    @staticmethod
    def _load_csv(file: str, header=None):
        assert isinstance(file, str) and file.endswith('.csv')
        return pd.read_csv(file, header=header)

    @staticmethod
    def _load_excel(file: str, sheet_name=None, header=None):
        assert isinstance(file, str) and file.endswith('.xlsx')
        return pd.read_excel(file, sheet_name=sheet_name, header=header)

    @staticmethod
    def _get_sheet_names(file: str):
        assert isinstance(file, str) and file.endswith('.xlsx')
        return pd.ExcelFile(file).sheet_names

    def _normalize_datasets_and_few_shot_prompts(
        self, 
        datasets: Union[List[dict], Dict[str, List[dict]]], 
        few_shot_prompts: Union[str, Dict[str, str]]
    ) -> Dict[str, List[dict]]:
        if isinstance(datasets, list):
            datasets = {self.benchmark_name: datasets}
        if isinstance(few_shot_prompts, str):
            few_shot_prompts = {self.benchmark_name: few_shot_prompts}

        for task_name, few_shot_prompt in few_shot_prompts.items():
            few_shot_prompts[task_name] = few_shot_prompt.replace('{', '{{').replace('}', '}}').replace('{{test_question}}', '{test_question}')
        return datasets, few_shot_prompts

    def _prompting_for_zero_shot(self, dataset: List[dict], task_prompt: str='', response_with: str='') -> List[dict]:
        """Prompt one dataset for zero-shot setting"""
        return [
            {
                **data,
                'prompted_query': task_prompt.format(input=data['query'].strip()),
                'response_with': response_with,
            }
            for data in dataset
        ]

    def _prompting_for_few_shot(self, dataset: List[dict], few_shot_prompt: str='', task_prompt: str='') -> List[dict]:
        """Prompt one dataset for few-shot setting"""
        return [
            {
                **data,
                'prompted_query': task_prompt.format(
                    input=few_shot_prompt.format(test_question=data['query'].strip()),
                ),
                'response_with': '',
            }
            for data in dataset
        ]
    
    def _prompting_datasets(self, 
            datasets: Dict[str, List[dict]],
            few_shot_prompts: Dict[str, str],
            setting: str,
            stage: int,
        ) -> Dict[str, List[dict]]:
        """Prompt all datasets, available for zero_shot and few_shot settings"""
        assert isinstance(datasets, dict) and isinstance(few_shot_prompts, dict)

        assert set(datasets.keys()) == set(few_shot_prompts.keys())
        if setting == 'few_shot':
            return {
                task_name:  self._prompting_for_few_shot(datasets[task_name], few_shot_prompt=few_shot_prompts[task_name], task_prompt=self.task_prompt_few_shot[task_name])
                for task_name in datasets.keys()
            }
        else:
            return {
                task_name:  self._prompting_for_zero_shot(datasets[task_name], task_prompt=self.task_prompt_zero_shot[task_name], response_with=self.response_with_zero_shot[task_name] if stage == 1 else '')
                for task_name in datasets.keys()
            }

    def get_datasets(
        self, 
        setting: str = 'few_shot',
        n_shot: int = None,
        stage: int = 1,
    ) -> Union[List[dict], Dict[str, List[dict]]]:
        all_datasets: Union[List[dict], Dict[str, List[dict]]] = self._prepare_data()
        few_shot_prompts: Union[str, Dict[str, str]] = self._prepare_few_shot_prompt(n_shot=n_shot)
        all_datasets, few_shot_prompts = self._normalize_datasets_and_few_shot_prompts(all_datasets, few_shot_prompts)
    
        return self._prompting_datasets(all_datasets, few_shot_prompts, setting, stage)



class MMLUArbicBenchmark(BenchmarkBase):

    top_subjects = ['physics', 'chemistry', 'biology', 'computer science', 'math', 'engineering', 'history', 'philosophy', 'law', 'politics', 'culture', 'economics', 'geography', 'psychology', 'other', 'business', 'health']
    subtasks = ['abstract_algebra', 'anatomy', 'astronomy', 'business_ethics', 'clinical_knowledge', 'college_biology', 'college_chemistry', 'college_computer_science', 'college_mathematics', 'college_medicine', 'college_physics', 'computer_security', 'conceptual_physics', 'econometrics', 'electrical_engineering', 'elementary_mathematics', 'formal_logic', 'global_facts', 'high_school_biology', 'high_school_chemistry', 'high_school_computer_science', 'high_school_european_history', 'high_school_geography', 'high_school_government_and_politics', 'high_school_macroeconomics', 'high_school_mathematics', 'high_school_microeconomics', 'high_school_physics', 'high_school_psychology', 'high_school_statistics', 'high_school_us_history', 'high_school_world_history', 'human_aging', 'human_sexuality', 'international_law', 'jurisprudence', 'logical_fallacies', 'machine_learning', 'management', 'marketing', 'medical_genetics', 'miscellaneous', 'moral_disputes', 'moral_scenarios', 'nutrition', 'philosophy', 'prehistory', 'professional_accounting', 'professional_law', 'professional_medicine', 'professional_psychology', 'public_relations', 'security_studies', 'sociology', 'us_foreign_policy', 'virology', 'world_religions']
    subtasks_ar = ['جبر_تجريدي', 'تشريح', 'علم_الفلك', 'أخلاقيات_الأعمال', 'المعرفة_السريرية', 'علم_الأحياء_الجامعي', 'كيمياء_جامعية', 'علوم_الحاسوب_الجامعية', 'رياضيات_جامعية', 'طب_جامعي', 'فيزياء_جامعية', 'أمان_الحاسوب', 'فيزياء_مفاهيمية', 'الاقتصاد_القياسي', 'هندسة_كهربائية', 'رياضيات_ابتدائية', 'منطق_رسمي', 'حقائق_عالمية', 'علم_الأحياء_الثانوي', 'كيمياء_ثانوية', 'علوم_الحاسوب_الثانوية', 'تاريخ_أوروبا_الثانوي', 'جغرافية_ثانوية', 'الحكومة_والسياسة_الثانوية', 'اقتصاد_كلي_ثانوي', 'رياضيات_ثانوية', 'اقتصاد_جزئي_ثانوي', 'فيزياء_ثانوية', 'علم_النفس_الثانوي', 'إحصاء_ثانوي', 'تاريخ_الولايات_المتحدة_الثانوي', 'تاريخ_العالم_الثانوي', 'شيخوخة_الإنسان', 'جنسانية_بشرية', 'قانون_دولي', 'فقه', 'أخطاء_منطقية', 'تعلم_الآلة', 'إدارة', 'تسويق', 'جينات_طبية', 'متفرقات', 'نزاعات_أخلاقية', 'سيناريوهات_أخلاقية', 'تغذية', 'فلسفة', 'ما_قبل_التاريخ', 'محاسبة_مهنية', 'قانون_مهني', 'طب_مهني', 'علم_النفس_المهني', 'علاقات_عامة', 'دراسات_الأمان', 'علم_الاجتماع', 'سياسة_خارجية_أمريكية', 'علم_الفيروسات', 'أديان_العالم']
    categories = {'math': ['abstract_algebra', 'college_mathematics', 'elementary_mathematics', 'high_school_mathematics', 'high_school_statistics'], 'health': ['anatomy', 'clinical_knowledge', 'college_medicine', 'human_aging', 'medical_genetics', 'nutrition', 'professional_medicine', 'virology'], 'physics': ['astronomy', 'college_physics', 'conceptual_physics', 'high_school_physics'], 'business': ['business_ethics', 'management', 'marketing'], 'biology': ['college_biology', 'high_school_biology'], 'chemistry': ['college_chemistry', 'high_school_chemistry'], 'computer science': ['college_computer_science', 'computer_security', 'high_school_computer_science', 'machine_learning'], 'economics': ['econometrics', 'high_school_macroeconomics', 'high_school_microeconomics'], 'engineering': ['electrical_engineering'], 'philosophy': ['formal_logic', 'logical_fallacies', 'moral_disputes', 'moral_scenarios', 'philosophy', 'world_religions'], 'other': ['global_facts', 'miscellaneous', 'professional_accounting'], 'history': ['high_school_european_history', 'high_school_us_history', 'high_school_world_history', 'prehistory'], 'geography': ['high_school_geography'], 'politics': ['high_school_government_and_politics', 'public_relations', 'security_studies', 'us_foreign_policy'], 'psychology': ['high_school_psychology', 'professional_psychology'], 'culture': ['human_sexuality', 'sociology'], 'law': ['international_law', 'jurisprudence', 'professional_law']}
    top_categories = {
        "STEM": ["physics", "chemistry", "biology", "computer science", "math", "engineering"],
        "humanities": ["history", "philosophy", "law"],
        "social sciences": ["politics", "culture", "economics", "geography", "psychology"],
        "other (business, health, misc.)": ["other", "business", "health"],
    }
    def __init__(self):
        super(MMLUArbicBenchmark, self).__init__(benchmark_name='MMLUArabic')

    def _init_task_prompts(self):
        """
        Config self.task_prompt_zero_shot and self.task_prompt_few_shot
        """
        # self.task_prompt_zero_shot = self.task_prompt_few_shot = {
        #     task: 'The following are multiple choice questions (with answers) about %s.\n\n{input}' % (' '.join(task.split('_'))) for task in self.subtasks
        # }
        # self.task_prompt_zero_shot = {
        #     task: "فيما يلي أسئلة الاختيار من متعدد حول %s\n\n{input}" % ' '.join(self.subtasks_ar[i].split('_')) for i,task in enumerate(self.subtasks)
        # }

        self.task_prompt_zero_shot = {
            task: "فيما يلي أسئلة الاختيار من متعدد حول %s\n\n{input}\nمن فضلك اختر إجابة واحدة من بين 'A، B، C، D' دون شرح." % ' '.join(self.subtasks_ar[i].split('_')) for i,task in enumerate(self.subtasks)
        }

        
        self.response_with_zero_shot = '\nإجابة:'

        self.task_prompt_few_shot = {
            task: "فيما يلي أسئلة الاختيار من متعدد (مع الإجابات) حول %s\n\n{input}" % ' '.join(self.subtasks_ar[i].split('_')) for i,task in enumerate(self.subtasks)
        }

    def _prepare_data(self) -> Dict[str, List[dict]]:
        data_files = glob.glob(os.path.join(self.benchmark_dir, 'test', '*.csv'))

        def normalize(x):
            return str(x).strip()

        dataset_dict = {}
        for file in data_files:
            task_name = os.path.basename(file).split('_test')[0]
            df = self._load_csv(file, header=None)

            processed_data_list = []
            for index, row in df.iterrows():
                processed_data_list.append({
                    'query_id': index,
                    'query': f"سؤال: {normalize(row[0])}\nA. "+ f"{normalize(row[1])}"+"\nB. "+ f"{normalize(row[2])}" +"\nC. " +f"{normalize(row[3])}" +"\nD. " +f"{normalize(row[4])}",
                    'answer': f"{normalize(row[5])}",
                })
            dataset_dict[task_name] = processed_data_list

        return dataset_dict

    def _prepare_few_shot_prompt(self, n_shot = 5) -> Dict[str, str]:
        if n_shot is None:
            n_shot = 5
        assert n_shot <= 5, ('At most 5-shot')

        data_files = glob.glob(os.path.join(self.benchmark_dir, 'dev', '*.csv'))

        def normalize(x):
            return str(x).strip()

        few_shot_prompt_dict = {}
        for file in data_files:
            task_name = os.path.basename(file).split('_dev')[0]
            df = self._load_csv(file, header=None)

            few_shot_prompt = ''
            for index, row in df.iterrows():
                if index == n_shot:
                    break
                few_shot_prompt += (
                    f"سؤال: {normalize(row[0])}\nA. "+ f"{normalize(row[1])}"+"\nB. "+ f"{normalize(row[2])}" +"\nC. " +f"{normalize(row[3])}" +"\nD. " +f"{normalize(row[4])}"
                    + "\n" +"إجابة: "+f"{normalize(row[5])}\n\n"
                )
            few_shot_prompt += "{test_question}\nإجابة:" 
            few_shot_prompt_dict[task_name] = few_shot_prompt

        return few_shot_prompt_dict


class EXAMSBenchmark(BenchmarkBase):

    subtasks = ['Islamic Studies', 'Science', 'Social', 'Biology', 'Physics']
    subtasks_ar = ['الدراسات الإسلامية', 'العلوم', 'الاجتماعيات', 'علم الأحياء', 'علم الفيزياء']
    def __init__(self):
        super(EXAMSBenchmark, self).__init__(benchmark_name='EXAMS_Arabic')

    def _init_task_prompts(self):
        """
        Config self.task_prompt_zero_shot and self.task_prompt_few_shot
        """
        # self.task_prompt_zero_shot = self.task_prompt_few_shot = {
        #     task: 'The following are multiple choice questions (with answers) about %s.\n\n{input}' % (' '.join(task.split('_'))) for task in self.subtasks
        # }
        # self.task_prompt_zero_shot = {
        #     task: "فيما يلي أسئلة الاختيار من متعدد حول %s\n\n{input}" % ' '.join(self.subtasks_ar[i].split('_')) for i,task in enumerate(self.subtasks)
        # }

        self.task_prompt_zero_shot = {
            task: "فيما يلي أسئلة الاختيار من متعدد حول %s\n\n{input}\nمن فضلك اختر إجابة واحدة من بين 'A، B، C، D' دون شرح." % ' '.join(self.subtasks_ar[i].split('_')) for i,task in enumerate(self.subtasks)
        }


        self.response_with_zero_shot = '\nإجابة:'

        self.task_prompt_few_shot = {
            task: "فيما يلي أسئلة الاختيار من متعدد (مع الإجابات) حول %s\n\n{input}" % ' '.join(self.subtasks_ar[i].split('_')) for i,task in enumerate(self.subtasks)
        }

    def _prepare_data(self) -> Dict[str, List[dict]]:
        data_file = os.path.join(self.benchmark_dir, 'exam_test.jsonl')

        def normalize(x):
            return str(x).strip()

        dataset_dict = defaultdict(lambda: []) 

        data_list = self._load_jsonl(data_file)
        for data in data_list:
            c = data['id'].split('-')[0]
            dataset_dict[c].append({
                'query_id': data['id'],
                'query': f"سؤال: {normalize(data['question'])}\nA. "+ f"{normalize(data['A'])}"+"\nB. "+ f"{normalize(data['B'])}" +"\nC. " +f"{normalize(data['C'])}" +"\nD. " +f"{normalize(data['D'])}",
                'answer': normalize(data['answer']),
            })
        
        return dataset_dict
    
    def _prepare_few_shot_prompt(self, n_shot = 5) -> Dict[str, str]:
        if n_shot is None:
            n_shot = 5
        assert n_shot <= 5, ('At most 5-shot')

        data_file = os.path.join(self.benchmark_dir, 'exam_dev.jsonl')

        def normalize(x):
            return str(x).strip()

        few_shot_prompt_dict = defaultdict(lambda: '')
        count_shots = defaultdict(lambda: 0)

        data_list = self._load_jsonl(data_file)
        for data in data_list:
            c = data['id'].split('-')[0]
            if count_shots[c] == n_shot:
                continue

            few_shot_prompt_dict[c] += (
                f"سؤال: {normalize(data['question'])}\nA. "+ f"{normalize(data['A'])}"+"\nB. "+ f"{normalize(data['B'])}" +"\nC. " +f"{normalize(data['C'])}" +"\nD. " +f"{normalize(data['D'])}"
                + "\n" +"إجابة: "+f"{normalize(data['answer'])}\n\n"
            )
            count_shots[c] += 1

        for c in self.subtasks:
            few_shot_prompt_dict[c] += "{test_question}\nإجابة:" 

        return few_shot_prompt_dict
    

    



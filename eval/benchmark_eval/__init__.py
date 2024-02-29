from benchmark_eval.evaluation import (
    MMLUArabicEvaluation,
    EXAMSEvaluation,
    ArabicCultureEvaluation
)


benchmark2class = {
    'MMLUArabic': MMLUArabicEvaluation,
    'EXAMS_Arabic': EXAMSEvaluation,
    'ArabicCulture': ArabicCultureEvaluation
}



CACHE_DATASET_NAME = 'my_dataset'

class Config():
    def __init__(self):
        self.retriever_name_or_path = "klue/roberta-base"
        self.generator_name_or_path = 'SKT-AI/KoBART'

        self.num_retriever = 1
        self.extractor_top_k = [15]


        # Retriever
        self.max_retrieval_len = 512
        self.max_source_len = 300
        self.max_chunks = 300


        # Generator
        self.beam_size = 5
        self.min_length = 100
        self.min_length = 50
        self.max_target_len = 150


        self.dataset = ['data/ko']


        # cache file of dataset
        self.cached_dataset = CACHE_DATASET_NAME
        self.overwrite_cache = False




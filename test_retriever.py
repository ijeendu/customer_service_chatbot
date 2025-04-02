from utils import generate_embedding, load_db_collection
from retriever import get_closest_match_from_db
from config import (
    CHROMA_DB_PATH,
    COLLECTION_NAME,
    NUM_RESULTS,
    OLLAMA_EMBEDDING_MODEL,
    USER_QUERY)

from deepeval.metrics import (
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    AnswerRelevancyMetric,
    FaithfulnessMetric
)

from deepeval.test_case import LLMTestCase
from deepeval.dataset import EvaluationDataset

from deepeval import evaluate
from langfuse.decorators import observe
# from langfuse import langfuse_context




@observe
def main():

    # Load chroma db
    collection = load_db_collection(CHROMA_DB_PATH, COLLECTION_NAME, OLLAMA_EMBEDDING_MODEL)
    retrieval_context = get_closest_match_from_db(collection, OLLAMA_EMBEDDING_MODEL, USER_QUERY, NUM_RESULTS)
    #actual_output = generate(query, retrieval_context)  # Replace with your LLM function
    #actual_output = generate_embedding(USER_QUERY, OLLAMA_EMBEDDING_MODEL)

    # # concatenate list of docs into one context
    closest_docs = " ".join(retrieval_context)


    # create a TestCase
    test_case = LLMTestCase(
    input=USER_QUERY,
    retrieval_context=retrieval_context,
    expected_output=closest_docs,
    actual_output=closest_docs
    )

    #dataset = EvaluationDataset([test_case])
    # retriever metrics
    contextual_precision = ContextualPrecisionMetric()
    contextual_recall = ContextualRecallMetric()
    contextual_relevancy = ContextualRelevancyMetric()
    #generator metrics
    # answer_relevancy = AnswerRelevancyMetric(threshold=0.8)
    # faithfulness = FaithfulnessMetric()

    # evaluate
    evaluate(
        #dataset,
        test_cases=[test_case],
        metrics=[
            contextual_precision,
            contextual_recall,
            contextual_relevancy#,
            # answer_relevancy,
            # faithfulness
        ],
    )


if __name__ == "__main__":
    main()



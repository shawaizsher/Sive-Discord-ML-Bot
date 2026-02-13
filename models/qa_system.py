"""
Question Answering System
Answer questions based on provided context
"""

from transformers import pipeline


class QASystem:
    def __init__(self, model_name="deepset/roberta-base-squad2"):
        """
        Initialize Q&A system
        
        Args:
            model_name: HuggingFace model name for question answering
        """
        self.qa_pipeline = pipeline(
            "question-answering",
            model=model_name
        )
    
    def answer(self, question, context):
        """
        Answer a question based on context
        
        Args:
            question: Question string
            context: Context text containing the answer
            
        Returns:
            dict with answer, score, and position
        """
        result = self.qa_pipeline(
            question=question,
            context=context
        )
        
        return {
            'answer': result['answer'],
            'confidence': result['score'],
            'start': result['start'],
            'end': result['end']
        }
    
    def answer_multiple(self, questions, context):
        """
        Answer multiple questions from same context
        
        Args:
            questions: List of question strings
            context: Context text
            
        Returns:
            List of answers
        """
        return [self.answer(q, context) for q in questions]


# Example usage
if __name__ == "__main__":
    qa = QASystem()
    
    context = """
    Discord is a VoIP and instant messaging social platform. Users have the 
    ability to communicate with voice calls, video calls, text messaging, and 
    media and files in private chats or as part of communities called "servers". 
    It was launched in 2015 and has over 150 million monthly active users.
    """
    
    questions = [
        "What is Discord?",
        "When was Discord launched?",
        "How many users does Discord have?"
    ]
    
    for question in questions:
        result = qa.answer(question, context)
        print(f"Q: {question}")
        print(f"A: {result['answer']} (confidence: {result['confidence']:.2f})\n")

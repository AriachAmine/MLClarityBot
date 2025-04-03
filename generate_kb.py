import os
import re
import time
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Constants
ML_CONCEPTS = [
    # Learning Paradigms
    "Supervised Learning",
    "Unsupervised Learning",
    "Semi-Supervised Learning",
    "Reinforcement Learning",
    "Self-Supervised Learning",
    "Online Learning",
    "Meta-Learning",
    
    # Model Types
    "Classification",
    "Regression",
    "Clustering",
    "Anomaly Detection",
    "Ranking Models",
    "Multi-Label Classification",
    "Multi-Task Learning",
    
    # Bias & Variance
    "Overfitting",
    "Underfitting",
    "Bias-Variance Tradeoff",
    "Regularization",
    
    # Model Validation & Evaluation
    "Cross-Validation",
    "Train-Test Split",
    "Holdout Validation",
    "Bootstrapping",
    "Accuracy",
    "Precision",
    "Recall",
    "F1-Score",
    "Mean Squared Error (MSE)",
    "Root Mean Squared Error (RMSE)",
    "Mean Absolute Error (MAE)",
    "R-squared (R²)",
    "Log Loss",
    "AUC-ROC Curve",
    "Confusion Matrix",
    "Precision-Recall Curve",
    "Perplexity (for NLP)",
    
    # Feature Engineering & Selection
    "Feature Engineering Basics",
    "Feature Scaling",
    "Feature Selection",
    "Dimensionality Reduction",
    "Principal Component Analysis (PCA)",
    "t-Distributed Stochastic Neighbor Embedding (t-SNE)",
    "Linear Discriminant Analysis (LDA)",
    "Mutual Information",
    "Polynomial Features",
    "Sparse Representations",
    
    # Optimization & Training
    "Gradient Descent",
    "Stochastic Gradient Descent (SGD)",
    "Adam Optimizer",
    "Momentum-based Optimization",
    "Nesterov Accelerated Gradient (NAG)",
    "RMSprop",
    "Learning Rate Scheduling",
    "Batch Normalization",
    "Dropout Regularization",
    "Early Stopping",
    "Hyperparameter Tuning",
    "Bayesian Optimization",
    "Grid Search",
    "Random Search",
    
    # Machine Learning Algorithms
    "K-Nearest Neighbors (KNN) Concept",
    "Decision Trees",
    "Random Forest",
    "Gradient Boosting",
    "XGBoost",
    "LightGBM",
    "CatBoost",
    "Support Vector Machines (SVM)",
    "Naive Bayes",
    "Linear Regression",
    "Logistic Regression",
    "Lasso Regression",
    "Ridge Regression",
    "Elastic Net",
    "Neural Networks",
    "Deep Learning",
    "Convolutional Neural Networks (CNNs)",
    "Recurrent Neural Networks (RNNs)",
    "Long Short-Term Memory (LSTM)",
    "Gated Recurrent Units (GRU)",
    "Transformer Models",
    
    # Ensemble Learning
    "Bagging",
    "Boosting",
    "Stacking",
    "Blending",
    "Voting Classifiers",
    
    # Explainability & Interpretability
    "SHAP Values",
    "LIME (Local Interpretable Model-agnostic Explanations)",
    "Model Explainability",
    "Fairness in ML",
    
    # Data Processing & Feature Representation
    "One-Hot Encoding",
    "Label Encoding",
    "Ordinal Encoding",
    "TF-IDF",
    "Word Embeddings",
    "Word2Vec",
    "GloVe",
    "Tokenization",
    "Sentence Embeddings",
    "BERT Embeddings",
    
    # Probabilistic Models
    "Bayesian Networks",
    "Hidden Markov Models (HMMs)",
    "Gaussian Mixture Models (GMMs)",
    
    # Graph-Based Machine Learning
    "Graph Neural Networks (GNNs)",
    "Graph Convolutional Networks (GCNs)",
    "Graph Attention Networks (GATs)",
    
    # Generative Models
    "Generative Adversarial Networks (GANs)",
    "Variational Autoencoders (VAEs)",
    "Diffusion Models",
    
    # Reinforcement Learning
    "Markov Decision Processes (MDP)",
    "Q-Learning",
    "Deep Q Networks (DQN)",
    "Policy Gradient Methods",
    "Actor-Critic Methods",
    "Proximal Policy Optimization (PPO)",
    "Monte Carlo Tree Search",
    
    # Advanced Topics
    "Self-Attention",
    "Transformers in NLP",
    "Few-Shot Learning",
    "Zero-Shot Learning",
    "Contrastive Learning",
    "Neural Architecture Search (NAS)",
    "Federated Learning",
    "Edge AI",
    "Quantum Machine Learning",
]


KNOWLEDGE_BASE_DIR = "knowledge_base"

# Initialize Gemini API
def init_gemini_model():
    """Initialize and return a Gemini API model."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("WARNING: GEMINI_API_KEY not found in .env file.")
        return None
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        return model
    except Exception as e:
        print(f"Error initializing Gemini model: {e}")
        return None

def generate_ml_concept_explanation(concept, gemini_model=None, max_retries=3, initial_delay=3):
    """
    Generate an explanation for a given ML concept using the Gemini API.
    
    Args:
        concept (str): The ML concept to explain
        gemini_model: The initialized Gemini model (optional)
        max_retries (int): Maximum number of retries for 429 errors
        initial_delay (int): Initial delay in seconds before retrying
        
    Returns:
        str: The generated explanation
    """
    if gemini_model is None:
        gemini_model = init_gemini_model()
        if gemini_model is None:
            return "Error: Could not initialize Gemini model."
    
    prompt = f"""
    Act as an expert Machine Learning tutor creating a resource file. Your goal is to explain the concept: **{concept}**.

    **Instructions:**

    1.  **Target Audience:** Explain the concept clearly and simply, targeting someone relatively new to Machine Learning. Avoid unnecessary jargon; if used, explain it briefly.
    2.  **Structure (Use Markdown):** Organize the response using *exactly* these Markdown headings in this order:
        *   `## Definition`: Provide a concise, 1-2 sentence definition of the core idea.
        *   `## Explanation`: Elaborate on the definition (how it works, why it's important). Keep focused.
        *   `## Analogy`: Provide *one* simple, relatable real-world analogy. Clearly explain the connection.
        *   `## Diagram Suggestion`: If a basic diagram aids understanding, suggest *one type* (e.g., flowchart, comparison table, simple axes) and *briefly describe its key components/flow* relevant to "{concept}". If no simple diagram adds much value, explicitly state: "No specific diagram is essential for grasping the core concept."
    3.  **Content Focus:** Ensure the explanation is self-contained for "{concept}". The analogy should be easy to grasp. The diagram suggestion must be for a *basic* visual.
    4.  **Length:** Aim for a total length between 200 and 450 words.
    5.  **Tone:** Clear, helpful, and accessible.

    **Concept to Explain:** {concept}
    """
    
    retries = 0
    while retries <= max_retries:
        try:
            response = gemini_model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.7,
                    max_output_tokens=600,
                ),
            )
            return response.text
        except Exception as e:
            error_str = str(e).lower()
            # Check if it's a quota exceeded error (429)
            if "429" in error_str or "quota" in error_str or "rate limit" in error_str:
                if retries < max_retries:
                    delay = initial_delay * (2 ** retries)  # Exponential backoff
                    print(f"Rate limit exceeded for {concept}. Retrying in {delay} seconds... (Attempt {retries+1}/{max_retries})")
                    time.sleep(delay)
                    retries += 1
                    continue
                else:
                    return f"Error: Exceeded retry attempts for {concept} due to quota limits. Try again later."
            else:
                # For non-quota errors, return immediately
                return f"Error generating explanation: {e}"
    
    return f"Error: Failed to generate explanation for {concept} after {max_retries} attempts."

def generate_clean_filename(concept):
    """
    Generate a clean filename from a concept name.
    
    Args:
        concept (str): The ML concept name
        
    Returns:
        str: A cleaned filename (lowercase with underscores)
    """
    # Convert to lowercase
    filename = concept.lower()
    # Replace spaces and special characters with underscores
    filename = re.sub(r'[^a-z0-9]', '_', filename)
    # Replace multiple underscores with a single one
    filename = re.sub(r'_+', '_', filename)
    # Add .txt extension
    return f"{filename}.txt"

if __name__ == "__main__":
    # Create output directory
    os.makedirs(KNOWLEDGE_BASE_DIR, exist_ok=True)
    
    # Initialize Gemini model
    print("Initializing Gemini Model...")
    model = init_gemini_model()
    
    if model is None:
        print("Failed to initialize Gemini Model. Exiting.")
        exit(1)
    
    print("Successfully initialized Gemini Model.")
    
    # Process each concept
    for concept in ML_CONCEPTS:
        print(f"Generating explanation for: {concept}...")
        
        # Generate explanation
        explanation = generate_ml_concept_explanation(concept, model)
        
        # Check if generation was successful
        if explanation.startswith("Error:"):
            print(f"ERROR generating explanation for {concept}: {explanation}")
            continue
        
        # Generate clean filename
        filename = generate_clean_filename(concept)
        output_filepath = os.path.join(KNOWLEDGE_BASE_DIR, filename)
        
        # Save explanation to file
        try:
            with open(output_filepath, "w", encoding="utf-8") as f:
                f.write(explanation)
            print(f"Successfully generated and saved: {output_filepath}")
        except Exception as e:
            print(f"ERROR saving explanation for {concept}: {e}")

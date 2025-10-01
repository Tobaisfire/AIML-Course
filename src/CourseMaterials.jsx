import React, { useState } from 'react';
import { BookOpen, Code, Download, ChevronDown, ChevronRight, FileText, Video, CheckCircle } from 'lucide-react';

const CourseMaterials = () => {
  const [expandedModule, setExpandedModule] = useState(1);
  const [expandedweek, setExpandedweek] = useState(null);

  const courseStructure = {
    Module1: {
      title: "Module 1: Python & Data Fundamentals",
      duration: "6 hours",
      weeks: [
        {
          id: "s1",
          title: "week 1: Python Essentials for AI",
          duration: "3 hours",
          objectives: [
            "Python  Programming basics",
            "Master NumPy arrays and operations for numerical computing"
          ],
          timeline: [
            {  activity: "Introduction & Setup", content: "Course overview, environment setup (Anaconda/Colab), verify installations" },
            {  activity: "Python Basics", content: "Variables, Data Types, Control Flow, Functions, Classes, Modules, Error Handling" },
            {  activity: "NumPy Deep Dive", content: "Arrays, indexing, slicing, broadcasting, mathematical operations" },
            {  activity: "Q&A & Assignment", content: "Doubt clearing, assign homework on housing dataset" },
            {  activity: "Wrap-up", content: "Key takeaways, preview next week" }
          ],
          codeTopics: [
            "print(), type(), len(), input(), if-else, for-loop, while-loop, functions, classes, modules, error handling",
            "import numpy as np, pandas as pd",
            "np.array(), array operations, reshaping",
           
          ],
          datasets: ["No dataset Needed"],
          assignments: "Mini Project: Based on the concepts learned in the week, create a program to solve a real-world problem"
        },  
        {
          id: "s2",
          title: "week 2: Data Visualization & Data Preprocessing",
          duration: "3 hours",
          objectives: [ "Learn Pandas DataFrames for data manipulation",
            "Understand data loading and basic exploration",
            "Apply skills to real-world dataset (Titanic)",
            "Create meaningful visualizations with Matplotlib & Seaborn",
            "Handle missing data effectively",
            "Encode categorical variables",
            "Prepare data for machine learning" ,
            "Understand the importance of data preprocessing"
          ],
          timeline: [
            {  activity: "Recap & Homework Review", content: "Quick review of week 1, discuss homework solutions" },
            {  activity: "Pandas Fundamentals", content: "Series, DataFrames, reading CSV, basic operations (head, describe, info)" },
            {  activity: "Hands-on Practice", content: "Load Titanic dataset, explore columns, check data types, basic statistics" },
            {  activity: "Mini Project", content: "Guided project: Analyze Titanic data - survival rates, age distribution, class analysis" },  
            {  activity: "Data Visualization", content: "Matplotlib basics, Seaborn for statistical plots, histograms, scatter plots, heatmaps" },
            {  activity: "Handling Missing Data", content: "Identify nulls, strategies (drop, mean/median imputation, forward fill)" },
            {  activity: "Feature Engineering", content: "Label encoding, one-hot encoding, feature scaling (StandardScaler, MinMaxScaler)" },
            {  activity: "Train-Test Split", content: "Concept of overfitting, train_test_split, stratification" },
            {  activity: "Mini Project", content: "Complete data preprocessing pipeline on messy dataset" },
          {  activity: "Wrap-up", content: "Summary, prepare for ML next Module" }
          ],
          codeTopics: [
            "pd.read_csv(), df.head(), df.info(), df.describe()",
            "Indexing: df['column'], df[['col1', 'col2']]",
            "Filtering: df[df['Age'] > 30]",
            "Groupby: df.groupby('Sex')['Survived'].mean()",
            "import matplotlib.pyplot as plt, seaborn as sns",
            "plt.hist(), plt.scatter(), sns.heatmap()",
            "df.isnull().sum(), df.fillna(), df.dropna()",  
            "pd.get_dummies(), LabelEncoder",
            "from sklearn.preprocessing import StandardScaler",
            "from sklearn.model_selection import train_test_split"
          ],
          datasets: ["Titanic (Kaggle)", "Housing Prices (optional homework)"],
          assignments: "Explore housing dataset: load data, find missing values, calculate average prices by location"
        }
      ]
    },
    Module2: {
      title: "Module 2: Machine Learning Basics",
      duration: "6 hours",
      weeks: [
        {
          id: "s3",
          title: "week 3: Supervised Learning - Regression and Supervised Learning - Classification ",
          duration: "3 hours",
          objectives: [
            "Understand regression problems and use cases",
            "Build and train Linear Regression models",
            "Evaluate model performance with metrics",
            "Save and load trained models",
            "Understand classification problems",
            "Build Logistic Regression and Decision Tree models",
            "Evaluate with confusion matrix and metrics",
            "Compare multiple models"
            
          ],
          timeline: [
            {  activity: "ML Introduction", content: "What is ML? Supervised vs Unsupervised, Regression vs Classification" },
            {  activity: "Linear Regression Theory", content: "Concept, equation, cost function (MSE), how it learns" },
            {  activity: "Scikit-learn Basics", content: "Import, fit, predict workflow, LinearRegression class" },
            {  activity: "Hands-on Project", content: "House price prediction: load data, train model, make predictions" },
            {  activity: "Model Evaluation", content: "MSE, RMSE, R² score, visualization of predictions vs actual" },
            {  activity: "Model Persistence", content: "Save with pickle/joblib, load and use saved model" },
            {  activity: "Practice & Q&A", content: "Students try different features, improve model" },
            {  activity: "Classification Introduction", content: "Binary vs multi-class, real-world examples" },
            {  activity: "Logistic Regression", content: "Concept, sigmoid function, implementation with sklearn" },
            {  activity: "Decision Trees", content: "How trees work, splits, Gini impurity, implementation" },
            {  activity: "Model Evaluation", content: "Accuracy, precision, recall, F1-score, confusion matrix, when to use which metric" },
            {  activity: "Project: Spam Detection", content: "Text classification (simplified), TF-IDF basics, build classifier" },
            {  activity: "Model Comparison", content: "Compare Logistic Regression vs Decision Tree, discuss results" }
          ],
          codeTopics: [
            "from sklearn.linear_model import LinearRegression",
            "model = LinearRegression()",
            "model.fit(X_train, y_train)",
            "predictions = model.predict(X_test)",
            "from sklearn.metrics import mean_squared_error, r2_score",
            "import joblib; joblib.dump(model, 'model.pkl')",
            "from sklearn.linear_model import LogisticRegression",
            "model = LogisticRegression()",
            "model.fit(X_train, y_train)",
            "predictions = model.predict(X_test)",
            "from sklearn.metrics import confusion_matrix, classification_report",
            "from sklearn.metrics import accuracy_score, precision_score",
            "from sklearn.feature_extraction.text import TfidfVectorizer",
            "import seaborn as sns; sns.heatmap(confusion_matrix)",
           
          ],
          datasets: ["Boston Housing", "California Housing","Spam SMS", "Customer Churn", "Iris Classification"],
          assignments: "Build regression model on salary prediction dataset, achieve R² > 0.7, Build spam detection model, achieve accuracy > 90%, Build customer churn predictor, achieve accuracy > 90%, Build iris classification model, achieve accuracy > 90%"
        },
        {
          id: "s4",
          title: "week 4: Ensemble Methods",
          duration: "3 hours",
          objectives: [
            "Understand ensemble learning concepts",
            "Implement Random Forest classifier",
            "Perform hyperparameter tuning",
            "Build end-to-end ML pipeline"
          ],
          timeline: [
            {  activity: "Ensemble Introduction", content: "Wisdom of crowds, bagging vs boosting" },
            {  activity: "Random Forest", content: "How it works, advantages, implementation" },
            {  activity: "Hyperparameter Tuning", content: "GridSearchCV, RandomizedSearchCV, cross-validation" },
            {  activity: "End-to-End Pipeline", content: "Complete ML project: data loading → preprocessing → training → evaluation → deployment" },
            {  activity: "Hands-on Project", content: "Credit card fraud detection or similar imbalanced dataset" },
            {  activity: "Best Practices", content: "Feature importance, avoiding overfitting, model selection tips" }
          ],
          codeTopics: [
            "from sklearn.ensemble import RandomForestClassifier",
            "from sklearn.model_selection import GridSearchCV, cross_val_score",
            "param_grid = {'n_estimators': [100, 200], 'max_depth': [10, 20]}",
            "grid_search = GridSearchCV(model, param_grid, cv=5)",
            "feature_importance = model.feature_importances_",
            "from sklearn.pipeline import Pipeline"
          ],
          datasets: ["Credit Card Fraud", "Wine Quality"],
          assignments: "Create complete ML pipeline on new dataset with proper validation"
        },

      ]
    },
    Module3: {
      title: "Module 3: Deep Learning Introduction & Practical",
      duration: "6 hours",
      weeks: [
        {
          id: "s5",
          title: "week 5: Deep Learning Introduction",
          duration: "2 hours",
          objectives: [
            "Understand neural network basics (high-level)",
            "Learn Keras/TensorFlow fundamentals",
            "Build first neural network",
            "Classify MNIST digits"
          ],
          timeline: [
            {  activity: "Neural Networks Introduction", content: "Biological inspiration, perceptron, layers, activations (no deep math)" },
            {  activity: "Keras Setup", content: "Installation, Sequential API, layers, compilation" },
            {  activity: "Building First Network", content: "Dense layers, activation functions, model.compile(), model.fit()" },
            {  activity: "MNIST Project", content: "Load MNIST, build CNN (simple), train, evaluate" },
            {  activity: "Understanding Training", content: "Epochs, batch size, loss curves, overfitting visualization" },
            {  activity: "Practice", content: "Students modify architecture, experiment with parameters" }
          ],
          codeTopics: [
            "import tensorflow as tf",
            "from tensorflow.keras.models import Sequential",
            "from tensorflow.keras.layers import Dense, Flatten, Conv2D",
            "model = Sequential([Dense(128, activation='relu'), Dense(10, activation='softmax')])",
            "model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')",
            "model.fit(X_train, y_train, epochs=10, validation_split=0.2)"
          ],
          datasets: ["MNIST Digits", "Fashion MNIST"],
          assignments: "Build neural network for Fashion MNIST, achieve >85% accuracy"
        },
  
        {
          id: "s6",
          title: "week 6: Computer Vision with CNNs",
          duration: "2 hours",
          objectives: [
            "Understand CNNs for image processing",
            "Learn transfer learning with pre-trained models",
            "Build custom image classifier",
            "Deploy with Gradio interface"
          ],
          timeline: [
            {  activity: "CNN Architecture", content: "Convolutional layers, pooling, filters (visual explanation)" },
            {  activity: "Transfer Learning", content: "Pre-trained models (ResNet, VGG, MobileNet), fine-tuning" },
            {  activity: "Image Classifier Project", content: "Cats vs Dogs or custom dataset, data augmentation" },
            {  activity: "Training & Optimization", content: "Callbacks, early stopping, model checkpoints" },
            {  activity: "Deployment Demo", content: "Create Gradio interface for image upload and prediction" },
            {  activity: "Showcase", content: "Students demo their classifiers" }
          ],
          codeTopics: [
            "from tensorflow.keras.applications import ResNet50, MobileNetV2",
            "base_model = ResNet50(weights='imagenet', include_top=False)",
            "from tensorflow.keras.preprocessing.image import ImageDataGenerator",
            "datagen = ImageDataGenerator(rotation_range=20, horizontal_flip=True)",
            "from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint",
            "import gradio as gr"
          ],
          datasets: ["Cats vs Dogs", "Plant Disease", "Custom dataset (students choice)"],
          assignments: "Build image classifier for any domain (5+ classes), create Gradio demo"
        },
        {
          id: "s8",
          title: "week 7: NLP Basics",
          duration: "2 hours",
          objectives: [
            "Learn text preprocessing techniques",
            "Understand word embeddings",
            "Build sentiment analyzer",
            "Use pre-trained NLP models"
          ],
          timeline: [
            {  activity: "NLP Introduction", content: "Text challenges, tokenization, why NLP is hard" },
            {  activity: "Text Preprocessing", content: "Cleaning, tokenization, stopwords, stemming/lemmatization" },
            {  activity: "Word Embeddings", content: "Word2Vec, GloVe concepts, using pre-trained embeddings" },
            {  activity: "Sentiment Analysis Project", content: "Movie reviews, LSTM/GRU basics, build classifier" },
            {  activity: "Pre-trained Models", content: "HuggingFace transformers intro, using BERT for classification" },
            {  activity: "Practice", content: "Students build sentiment analyzer for different domain" }
          ],
          codeTopics: [
            "import nltk; from nltk.tokenize import word_tokenize",
            "from nltk.corpus import stopwords",
            "from tensorflow.keras.layers import LSTM, Embedding",
            "from tensorflow.keras.preprocessing.text import Tokenizer",
            "from tensorflow.keras.preprocessing.sequence import pad_sequences",
            "from transformers import pipeline"
          ],
          datasets: ["IMDB Movie Reviews", "Twitter Sentiment", "Product Reviews"],
          assignments: "Build sentiment analyzer for product reviews, compare LSTM vs transformer"
        }
      ]
    },
    Module5: {
      title: "Module 4: GenAI - LLMs & Prompt Engineering",
      duration: "6 hours",
      weeks: [
        {
          id: "s7",
          title: "week 9: Introduction to LLMs",
          duration: "3 hours",
          objectives: [
            "Understand what LLMs are and how they work",
            "Learn to use LLM APIs effectively",
            "Master prompt engineering techniques",
            "Build applications with different LLMs"
          ],
          timeline: [
            {  activity: "LLM Landscape", content: "GPT, Claude, Llama, Gemini - capabilities and differences" },
            {  activity: "API Setup", content: "OpenAI/Anthropic API keys, pricing, rate limits, basic usage" },
            {  activity: "Prompt Engineering", content: "Zero-shot, few-shot, chain-of-thought, system prompts, examples" },
            {  activity: "Parameters & Control", content: "Temperature, top_p, max_tokens, stop sequences" },
            {  activity: "Hands-on Projects", content: "Build: summarizer, translator, code generator, data extractor" },
            {  activity: "Best Practices", content: "Prompt templates, error handling, cost optimization" }
          ],
          codeTopics: [
            "import openai; from anthropic import Anthropic",
            "response = openai.ChatCompletion.create(model='gpt-4', messages=[...])",
            "System prompts, user prompts, assistant prompts",
            "Few-shot examples in prompts",
            "Temperature control for creativity vs consistency",
            "Streaming responses, async calls"
          ],
          datasets: ["Sample documents for summarization", "Code snippets for explanation"],
          assignments: "Create 5 different prompt templates for various tasks, test and optimize them"
        },
        {
          id: "s8",
          title: "week 10: HuggingFace & Model Usage",
          duration: "3 hours",
          objectives: [
            "Navigate HuggingFace Hub",
            "Use pre-trained models for various tasks",
            "Build practical NLP applications",
            "Deploy models locally"
          ],
          timeline: [
            {  activity: "HuggingFace Ecosystem", content: "Hub, Models, Datasets, Spaces overview" },
            {  activity: "Transformers Library", content: "Pipeline API, AutoModel, AutoTokenizer" },
            {  activity: "Text Generation", content: "GPT-2, GPT-Neo, Llama models (if available)" },
            {  activity: "Specialized Tasks", content: "Summarization (BART, T5), Translation (MarianMT), QA (BERT)" },
            {  activity: "Project: Multi-tool App", content: "Build Streamlit app with summarizer, translator, Q&A" },
            {  activity: "Model Selection", content: "Choosing right model for task, size vs performance tradeoffs" }
          ],
          codeTopics: [
            "from transformers import pipeline, AutoModel, AutoTokenizer",
            "summarizer = pipeline('summarization', model='facebook/bart-large-cnn')",
            "translator = pipeline('translation_en_to_fr', model='Helsinki-NLP/opus-mt-en-fr')",
            "generator = pipeline('text-generation', model='gpt2')",
            "import streamlit as st",
            "Model quantization for faster inference"
          ],
          datasets: ["News articles", "Wikipedia pages", "Technical documentation"],
          assignments: "Build text processing app with 3+ features using HuggingFace models"
        }
      ]
    },
    Module6: {
      title: "Module 5: GenAI - RAG & LangChain",
      duration: "6 hours",
      weeks: [
        {
          id: "s9",
          title: "week 11: LangChain Fundamentals",
          duration: "3 hours",
          objectives: [
            "Understand LangChain architecture",
            "Build chains and agents",
            "Implement conversational memory",
            "Create chatbot applications"
          ],
          timeline: [
            {  activity: "LangChain Introduction", content: "Why LangChain? Components: prompts, chains, agents, memory" },
            {  activity: "Prompts & Chains", content: "PromptTemplate, LLMChain, SequentialChain" },
            {  activity: "Memory Systems", content: "ConversationBufferMemory, ConversationSummaryMemory" },
            {  activity: "Building Chatbot", content: "Create conversational agent with context retention" },
            {  activity: "Agents & Tools", content: "ReAct agents, custom tools, agent executors" },
            {  activity: "Demo & Discussion", content: "Students showcase chatbots, discuss improvements" }
          ],
          codeTopics: [
            "from langchain import PromptTemplate, LLMChain",
            "from langchain.chat_models import ChatOpenAI",
            "from langchain.memory import ConversationBufferMemory",
            "from langchain.chains import ConversationChain",
            "from langchain.agents import initialize_agent, Tool",
            "Streaming responses in LangChain"
          ],
          datasets: ["Custom conversation data", "FAQ datasets"],
          assignments: "Build specialized chatbot for specific domain (customer service, tutor, etc.)"
        },
        {
          id: "s10",
          title: "week 12: RAG & Final Project",
          duration: "3 hours",
          objectives: [
            "Understand Retrieval Augmented Generation",
            "Work with vector databases",
            "Build document Q&A system",
            "Complete capstone project"
          ],
          timeline: [
            {  activity: "RAG Concept", content: "Why RAG? Limitations of LLMs, grounding in documents" },
            {  activity: "Vector Databases", content: "Embeddings, ChromaDB/FAISS, similarity search" },
            {  activity: "Document Processing", content: "Loading PDFs, chunking strategies, creating embeddings" },
            {  activity: "Final Project Build", content: "PDF chatbot: upload docs, ask questions, get answers with sources" },
            {  activity: "Testing & Optimization", content: "Improve retrieval, chunk sizing, prompt tuning" },
            {  activity: "Course Wrap-up", content: "Student demos, next steps, resources for continued learning" }
          ],
          codeTopics: [
            "from langchain.document_loaders import PyPDFLoader",
            "from langchain.text_splitter import RecursiveCharacterTextSplitter",
            "from langchain.embeddings import OpenAIEmbeddings",
            "from langchain.vectorstores import Chroma, FAISS",
            "from langchain.chains import RetrievalQA",
            "retriever = vectorstore.as_retriever()"
          ],
          datasets: ["Research papers", "Company documents", "Technical manuals"],
          assignments: "Final Project: Complete document Q&A system with UI (Streamlit/Gradio)"
        }
      ]
    }
  };

  const toggleModule = (Module) => {
    setExpandedModule(expandedModule === Module ? null : Module);
  };

  const toggleweek = (weekId) => {
    setExpandedweek(expandedweek === weekId ? null : weekId);
  };

  return (
    <div className="min-h-screen w-screen bg-gradient-to-br from-blue-50 to-indigo-100 p-8">
      <div className="max-w-7xl mx-auto">
        <div className="bg-white rounded-2xl shadow-2xl p-8 mb-8">
          <div className="flex items-center gap-4 mb-4">
            <BookOpen className="w-12 h-12 text-indigo-600" />
            <div>
              <h1 className="text-4xl font-bold text-gray-800">AI/ML Practical Course</h1>
              <p className="text-gray-600">30-Hour Comprehensive Program - BCA Students</p>
            </div>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-6">
            <div className="bg-indigo-50 p-4 rounded-lg">
              <h3 className="font-semibold text-indigo-800">Duration</h3>
              <p className="text-2xl font-bold text-indigo-600">30 Hours</p>
            </div>
            <div className="bg-green-50 p-4 rounded-lg">
              <h3 className="font-semibold text-green-800">Projects</h3>
              <p className="text-2xl font-bold text-green-600">7 Hands-on</p>
            </div>
            <div className="bg-purple-50 p-4 rounded-lg">
              <h3 className="font-semibold text-purple-800">Format</h3>
              <p className="text-2xl font-bold text-purple-600">70% Practical</p>
            </div>
          </div>
        </div>

        {Object.entries(courseStructure).map(([ModuleKey, Module], ModuleIndex) => (
          <div key={ModuleKey} className="bg-white rounded-xl shadow-lg mb-6 overflow-hidden">
            <div 
              className="bg-gradient-to-r from-indigo-500 to-purple-500 p-6 cursor-pointer flex justify-between items-center"
              onClick={() => toggleModule(ModuleIndex + 1)}
            >
              <div className="flex items-center gap-4">
                {expandedModule === ModuleIndex + 1 ? 
                  <ChevronDown className="w-6 h-6 text-white" /> : 
                  <ChevronRight className="w-6 h-6 text-white" />
                }
                <div>
                  <h2 className="text-2xl font-bold text-white">{Module.title}</h2>
                  <p className="text-indigo-100">{Module.duration}</p>
                </div>
              </div>
            </div>

            {expandedModule === ModuleIndex + 1 && (
              <div className="p-6">
                {Module.weeks.map((week) => (
                  <div key={week.id} className="mb-6 border-l-4 border-indigo-300 pl-6">
                    <div 
                      className="cursor-pointer flex items-center justify-between mb-4 hover:bg-gray-50 p-3 rounded"
                      onClick={() => toggleweek(week.id)}
                    >
                      <div className="flex items-center gap-3">
                        <Code className="w-6 h-6 text-indigo-600" />
                        <div>
                          <h3 className="text-xl font-semibold text-gray-800">{week.title}</h3>
                          <p className="text-sm text-gray-600">{week.duration}</p>
                        </div>
                      </div>
                      {expandedweek === week.id ? 
                        <ChevronDown className="w-5 h-5 text-gray-600" /> : 
                        <ChevronRight className="w-5 h-5 text-gray-600" />
                      }
                    </div>

                    {expandedweek === week.id && (
                      <div className="mt-4 space-y-6">
                        <div className="bg-blue-50 p-4 rounded-lg">
                          <h4 className="font-semibold text-blue-900 mb-3 flex items-center gap-2">
                            <CheckCircle className="w-5 h-5" />
                            Learning Objectives
                          </h4>
                          <ul className="space-y-2">
                            {week.objectives.map((obj, idx) => (
                              <li key={idx} className="text-blue-800 flex items-start gap-2">
                                <span className="text-blue-600 mt-1">•</span>
                                <span>{obj}</span>
                              </li>
                            ))}
                          </ul>
                        </div>

                        <div className="bg-green-50 p-4 rounded-lg">
                          <h4 className="font-semibold text-green-900 mb-3 flex items-center gap-2">
                            <Video className="w-5 h-5" />
                            week Timeline
                          </h4>
                          <div className="space-y-3">
                            {week.timeline.map((item, idx) => (
                              <div key={idx} className="border-l-2 border-green-300 pl-4">
                                <div className="flex items-baseline gap-2">
                                  <span className="font-semibold text-green-700 text-sm">{item.time}</span>
                                  <span className="font-medium text-green-900">{item.activity}</span>
                                </div>
                                <p className="text-green-800 text-sm mt-1">{item.content}</p>
                              </div>
                            ))}
                          </div>
                        </div>

                        <div className="bg-purple-50 p-4 rounded-lg">
                          <h4 className="font-semibold text-purple-900 mb-3 flex items-center gap-2">
                            <Code className="w-5 h-5" />
                            Key Code Topics
                          </h4>
                          <div className="bg-gray-900 text-green-400 p-4 rounded font-mono text-sm">
                            {week.codeTopics.map((code, idx) => (
                              <div key={idx} className="mb-2">{code}</div>
                            ))}
                          </div>
                        </div>

                        <div className="bg-yellow-50 p-4 rounded-lg">
                          <h4 className="font-semibold text-yellow-900 mb-3 flex items-center gap-2">
                            <FileText className="w-5 h-5" />
                            Datasets & Resources
                          </h4>
                          <div className="flex flex-wrap gap-2">
                            {week.datasets.map((dataset, idx) => (
                              <span key={idx} className="bg-yellow-200 text-yellow-900 px-3 py-1 rounded-full text-sm">
                                {dataset}
                              </span>
                            ))}
                          </div>
                        </div>

                        <div className="bg-red-50 p-4 rounded-lg">
                          <h4 className="font-semibold text-red-900 mb-2 flex items-center gap-2">
                            <Download className="w-5 h-5" />
                            Assignment
                          </h4>
                          <p className="text-red-800">{week.assignments}</p>
                        </div>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>
        ))}

        <div className="bg-gradient-to-r from-green-500 to-teal-500 rounded-xl shadow-lg p-8 text-white">
          <h2 className="text-3xl font-bold mb-4">Course Highlights</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <h3 className="text-xl font-semibold mb-3">✨ Key Projects</h3>
              <ul className="space-y-2">
                <li>📊 House Price Predictor (ML)</li>
                <li>🔔 Customer Churn Classifier</li>
                <li>🖼️ Image Classification System</li>
                <li>💬 Sentiment Analysis Tool</li>
                <li>📝 Text Summarizer</li>
                <li>🤖 Conversational Chatbot</li>
                <li>📚 Document Q&A System (RAG)</li>
              </ul>
            </div>
            <div>
              <h3 className="text-xl font-semibold mb-3">🛠️ Technologies Covered</h3>
              <ul className="space-y-2">
                <li>Python, NumPy, Pandas</li>
                <li>Scikit-learn, TensorFlow/Keras</li>
                <li>HuggingFace Transformers</li>
                <li>LangChain, Vector Databases</li>
                <li>OpenAI/Anthropic APIs</li>
                <li>Streamlit/Gradio for deployment</li>
              </ul>
            </div>
          </div>
        </div>

        <div className="mt-8 bg-white rounded-xl shadow-lg p-8">
          <h2 className="text-2xl font-bold text-gray-800 mb-4">Teaching Resources Included</h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="border-2 border-indigo-200 rounded-lg p-4 hover:border-indigo-400 transition">
              <h3 className="font-semibold text-indigo-700 mb-2">📓 Jupyter Notebooks</h3>
              <p className="text-gray-600 text-sm">Complete code for all 12 weeks with detailed explanations</p>
            </div>
            <div className="border-2 border-green-200 rounded-lg p-4 hover:border-green-400 transition">
              <h3 className="font-semibold text-green-700 mb-2">📊 Datasets</h3>
              <p className="text-gray-600 text-sm">Curated datasets for each project with download links</p>
            </div>
            <div className="border-2 border-purple-200 rounded-lg p-4 hover:border-purple-400 transition">
              <h3 className="font-semibold text-purple-700 mb-2">📝 Assignments</h3>
              <p className="text-gray-600 text-sm">Practice exercises and evaluation rubrics</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default CourseMaterials;
# IMDB Movie Review Sentiment Analysis

A comprehensive deep learning project that performs binary sentiment classification on IMDB movie reviews using TensorFlow/Keras. This implementation demonstrates fundamental deep learning concepts including overfitting, regularization techniques, and model optimization.

## Project Overview

This project builds and compares two neural network models:
- **Baseline Model**: Simple sequential network prone to overfitting
- **Regularized Model**: Enhanced with dropout, L2 regularization, and batch normalization

## Key Features

- **Complete Pipeline**: Data loading, preprocessing, model building, training, and evaluation
- **Overfitting Demonstration**: Clear visualization of training vs validation performance divergence
- **Regularization Techniques**: Implementation of dropout, L2 regularization, and batch normalization
- **Adaptive Training**: Early stopping and learning rate scheduling
- **Comprehensive Analysis**: Detailed performance comparison and visualization

## Dataset

- **Source**: IMDB Movie Review Dataset (built into Keras)
- **Size**: 50,000 reviews (25k train, 25k test)
- **Classes**: Binary (Positive/Negative sentiment)
- **Vocabulary**: Top 10,000 most frequent words
- **Sequence Length**: Padded to 500 tokens

## Model Architecture

### Baseline Model
```
Embedding(10000, 128) → GlobalAveragePooling1D → Dense(128) → Dense(64) → Dense(32) → Dense(1, sigmoid)
```

### Regularized Model
```
Embedding(10000, 128) → GlobalAveragePooling1D → 
Dense(128) + L2 + Dropout(0.5) + BatchNorm → 
Dense(64) + L2 + Dropout(0.4) + BatchNorm → 
Dense(32) + L2 + Dropout(0.3) → 
Dense(1, sigmoid)
```

## Results

| Metric | Baseline Model | Regularized Model | Improvement |
|--------|----------------|-------------------|-------------|
| Test Accuracy | 86.59% | 87.18% | +0.59% |
| Validation Loss | 0.4475 | 0.3428 | -23.4% |
| Overfitting Gap | 8.56% | 2.57% | -70.0% |

## Key Findings

1. **Regularization Impact**: Dropout, L2 regularization, and batch normalization significantly reduced overfitting
2. **Training Efficiency**: Early stopping prevented unnecessary training (14 vs 20 epochs)
3. **Generalization**: Regularized model showed better validation performance and stability
4. **Adaptive Strategies**: Learning rate scheduling improved convergence

## Code Structure

The implementation is organized into 7 distinct tasks:

1. **Task 1**: Dataset import and exploration
2. **Task 2**: Data preprocessing and train/validation split
3. **Task 3**: Sequential model architecture design
4. **Task 4**: Model compilation and training
5. **Task 5**: Training history visualization
6. **Task 6**: Regularization implementation
7. **Task 7**: Performance comparison and analysis

## Requirements

```
tensorflow>=2.0.0
numpy
matplotlib
scikit-learn
```

## Usage

Run each task sequentially in separate code cells:

```python
# Task 1: Import dataset
(X_train_full, y_train_full), (X_test, y_test) = imdb.load_data(num_words=10000)

# Task 2: Preprocess data
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2)

# Task 3-7: Follow the complete implementation
```

## Visualizations

The project includes comprehensive plots showing:
- Training vs validation loss curves
- Training vs validation accuracy progression  
- Performance comparison between models
- Clear demonstration of overfitting mitigation

## Educational Value

This project demonstrates:
- **Deep Learning Fundamentals**: Embedding layers, sequential models, activation functions
- **Overfitting Recognition**: How to identify and visualize overfitting patterns
- **Regularization Techniques**: Practical application of dropout, L2, and batch normalization
- **Model Optimization**: Early stopping, learning rate scheduling
- **Performance Evaluation**: Comprehensive model comparison methodology

## Future Improvements

- Implement LSTM/GRU layers for sequential processing
- Add attention mechanisms
- Experiment with pre-trained embeddings (GloVe, Word2Vec)
- Implement cross-validation
- Add more sophisticated regularization techniques

## License

MIT License - feel free to use for educational purposes.

---

**Note**: This implementation is optimized for Google Colab and educational demonstration of deep learning concepts.

# Sun-Sense: Accurate Solar Energy Projections With Deep Learning

## **Project Objective 🎯**

The project objective is to develop an accurate solar energy forecasting system using deep learning techniques. The primary goal is to provide reliable predictions of solar energy production based on historical data and relevant weather variables. The objective is to enable stakeholders in the solar energy industry, such as solar plant operators, grid managers, and renewable energy companies to make informed decisions, optimize resource allocation, and improve grid management by leveraging accurate forecasts.

## **Project Scope 👀**

The project aims to develop a deep learning-based solar energy forecasting system. The key elements of the project scope include:

1. **Data Collection**: Gathering historical solar energy production data from reliable sources, such as solar power plants, energy grid operators, or public databases. Acquiring relevant weather data, including irradiance, temperature, cloud cover, and other meteorological variables, from reputable weather stations or meteorological services.

2. **Data Preprocessing**: Cleaning and preprocessing the collected data to ensure its quality and compatibility with the deep learning models. This may involve handling missing values, normalizing data, and synchronizing timestamps between the solar energy and weather datasets.

3. **Model Development**: Designing and training deep learning models suitable for solar energy forecasting, such as RNNs, CNNs, or Transformers. The choice of models will be based on the nature of the data and the specific forecasting task.

4. **Computational Resources**: Assessing the computational resources required for model training and evaluation. This may include high-performance computing resources, such as GPUs (Graphics Processing Units), to expedite the training process and handle large-scale datasets.

5. **User Interface**: Developing a user-friendly interface that allows users to input relevant parameters, access the generated solar energy forecasts, and visualize the results in an intuitive manner. This may involve web development or software engineering skills, depending on the chosen platform.

6. **Documentation and Reporting**: Documenting the methodology, algorithms used, and guidelines for using the solar energy forecasting system. Creating comprehensive documentation to ensure transparency and facilitate future maintenance or updates.

## **Project Applications 🌏**

1. Solar Power Plants:
Optimize energy generation, enhance grid integration, and maximize profitability by accurately forecasting solar energy production, enabling precise planning of maintenance, dispatching strategies, and revenue optimization.

2. Solar System Owners:
Empower solar system owners to optimize their energy usage, increase self-consumption, and reduce reliance on the grid by providing accurate solar energy forecasts for effective energy management and demand-side planning.

3. Grid Operators:
Ensure grid stability, enhance load forecasting accuracy, and enable seamless integration of solar power into the grid by leveraging precise solar energy forecasts for load balancing, grid management, and facilitating the integration of renewable energy sources.

4. Energy Regulators and Policymakers:
Leverage accurate solar energy forecasting to inform renewable energy policies, drive sustainable energy planning, and enable effective decision-making for a more resilient, clean, and cost-efficient energy infrastructure.

## **Project Resources 📚**

 * Journal Article (reference): https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9737470
 * Project Dataset: https://power.larc.nasa.gov/
 * Extended Datasets: https://github.com/Ahmed-Sohail2000/Solar-Power-Datasets-and-Resources#solar-power-datasets-and-resources
 * Time Series Deep Learning: https://github.com/satellite-image-deep-learning/techniques#9-time-series
 * Data Analysis: https://www.simplilearn.com/data-analysis-methods-process-types-article
 * Data Preprocessing and Manipulation: https://www.xenonstack.com/blog/data-preprocessing-wrangling-ml
 * Data Visualization: https://www.tableau.com/learn/articles/data-visualization

 * PyTorch Forecasting Library: https://pytorch-forecasting.readthedocs.io/en/stable/getting-started.html

 * PyTorch Lightning Framework: https://www.pytorchlightning.ai/index.html

 * Time Series Forecasting Using Various Deep Learning Models: https://arxiv.org/ftp/arxiv/papers/2204/2204.11115.pdf

# **Chapter 1: Data Collection 🔢**

[Data Collection](https://www.google.com/url?sa=i&url=https%3A%2F%2Fwww.javatpoint.com%2Fdata-processing-in-data-mining&psig=AOvVaw34gvqe77SPssIcVodpA0Fz&ust=1686223846098000&source=images&cd=vfe&ved=0CBEQjRxqFwoTCJiikL6Hsf8CFQAAAAAdAAAAABAZ) is the process of gathering relevant data for deep learning projects. In deep learning, it is crucial to collect a large, diverse, and properly labeled dataset to train deep neural networks. This data allows the models to learn patterns, generalize well, and make accurate predictions. Data collection helps mitigate overfitting and enables the extraction of meaningful representations from the data. It is a critical step in deep learning to ensure the success and effectiveness of the models.

## **Where data is collected?**
The dataset that I have used for this project is **NASA Prediction of Worldwide Energy Resources** [Enhanced Data Access Beta Version](https://power.larc.nasa.gov/beta/data-access-viewer/) and country along with the region I have chosen is Sindh, Pakistan.

The regional statistics of solar resource and PVOUT are calculated from long-term daily averages based on the period from January 1st, 2015 - January 1st, 2023

## 1.5 **Data Visualization 📊**

![image](https://github.com/Ahmed-Sohail2000/Sun-Sense-Project/assets/107795296/2acec537-157d-46a0-9cf2-7b801f179ffa)

![image](https://github.com/Ahmed-Sohail2000/Sun-Sense-Project/assets/107795296/c0963727-4695-45c5-b572-baa76ca3d146)

![image](https://github.com/Ahmed-Sohail2000/Sun-Sense-Project/assets/107795296/99e23dad-7c72-457c-9a32-0adb3a404f3d)

![image](https://github.com/Ahmed-Sohail2000/Sun-Sense-Project/assets/107795296/98b0dfe0-4743-4e9e-a08b-2860d5f12b03)

# **Chapter 2: Modelling Experiments with Deep Learning 🖥️🧠🧮**

[Deep Learning](https://www.techtarget.com/searchenterpriseai/definition/deep-learning-deep-neural-network) is a subfield of machine learning that focuses on training artificial neural networks with multiple layers to learn and make predictions or decisions from complex data. Deep learning models are particularly effective for tasks such as image and speech recognition, natural language processing, and time series forecasting.

As for the framework, popular deep learning frameworks that provide efficient implementation of these models include:

- [TensorFlow](https://www.tensorflow.org/?gad=1&gclid=CjwKCAjw-b-kBhB-EiwA4fvKrEOkYZvsubeieqlLbUKcI1DdVVzibUBcnucxcM2ykv1m-j4XKXEy9xoCebwQAvD_BwE): An open-source library developed by Google that provides a wide range of tools and functionalities for building and training deep learning models.

- [Keras](https://keras.io/): A high-level deep learning library built on top of TensorFlow (and also available with other backends such as Theano and CNTK). Keras offers a user-friendly interface and allows for quick prototyping of deep learning models.

- [PyTorch](https://pytorch.org/): An open-source machine learning framework known for its flexibility and Pythonic syntax. It provides a platform for building and training deep neural networks, with features like dynamic computational graphs, GPU acceleration, automatic differentiation, and a rich ecosystem of libraries and tools.

Both TensorFlow and Keras provide comprehensive documentation, tutorials, and a large community support, making them suitable choices for your deep learning tasks. PyTorch is widely used for computer vision, natural language processing, and reinforcement learning tasks. It offers deployment options for various environments and has extensive documentation and community support.

Before carrying out the frameworks and models, there are 2 terms that are needed to familiarize in time series analysis and that is `horizon` and `window`.

* `Horizon`: Is the number of timesteps to predict/forecast into the future.
* `Window` : Is the number of past timesteps used to predict `Horizon`.

For example, if I want to forecast the next 6 months (180 days) of  solar irradiance values by using the past 1 year (365 days). Then the horizon would be 180 days and the window would be 365 days.

## 2.1 **Deep Learning Models 🤖🖥️👨🏻‍💻**

The model architecture that we would be implementing on our training data are listed below and some are going to be used for our data and evaluated whilst other models will not be selected based on the complexity and nature of our data.

1. Autoregressive Integrated Moving Average (ARIMA): ARIMA models are a classical approach for time series forecasting. They consider the autoregressive (AR), moving average (MA), and differencing (I) components of the time series to make predictions.

2. Long Short-Term Memory (LSTM) Networks: LSTM networks, a type of recurrent neural network (RNN), are effective for capturing long-term dependencies in time series data. They have a memory cell that can retain information over time and are suitable for modeling complex temporal patterns.

3. Gated Recurrent Unit (GRU) Networks: GRU networks are another type of RNN that can capture temporal dependencies in time series data. They have gating mechanisms that control the flow of information through the network and can be more computationally efficient than LSTMs.

4. Convolutional Neural Networks (CNNs): While CNNs are commonly used for image processing, they can also be applied to time series forecasting. By treating the time series as an image with one dimension, CNNs can extract features and learn patterns from the data.

5. Transformer Networks: Transformer networks have gained popularity in natural language processing tasks, but they can also be adapted for time series forecasting. Transformers leverage self-attention mechanisms to capture relationships between different time steps and have shown promising results in modeling sequential data.

6. Ensemble Models: Ensemble models combine multiple individual models to make predictions. For time series forecasting, ensemble models such as the combination of multiple ARIMA models or the combination of different LSTM models can improve overall accuracy and robustness.

It's important to experiment with different architectures and select the one that best suits specific time series data and forecasting requirements. Additionally, factors such as the complexity of your data, available computational resources, and interpretability of the model should also be considered when choosing an architecture.

## 3.7 **Model 6: Train a model on full historical data to make predictions into the future**

![image](https://github.com/Ahmed-Sohail2000/Sun-Sense-Project/assets/107795296/c0d0ced3-2f6d-4563-98f2-43756386de37)

## 3.8 **Compare Models**

![image](https://github.com/Ahmed-Sohail2000/Sun-Sense-Project/assets/107795296/f4095e0f-0144-404d-ab7b-0b83f6a7ffe6)

The majority of our deep learnign models are on par with the rest except for `model_3`, `naive_forecast`, and `model_2`. The best performing model is `model_6_full_data` followed by `model_6_multivariate`, `Conv1D`, `LSTM`, and `model_1`. The `ensemble_model` performed better than the `NBEATS` model which goes to show the complexity and size of layers does not always determine the suitability of that model for a certain type of data and how which model are better suited based on experimentation.

# 3.9 **Prophet AI 🧑‍🚀⌚**

[Prophet](https://facebook.github.io/prophet/) is an open-source forecasting tool developed by Facebook's Core Data Science team. It is designed to handle time series forecasting tasks and is particularly useful for business forecasting scenarios. Prophet is designed to be user-friendly and provide accurate and interpretable forecasts for various time series data, including those with missing values and irregularities.

**Key features of Prophet include:**

1) `Automatic Handling of Holidays and Special Events`: Prophet allows you to include holidays and other special events that might impact your time series data. It can automatically identify and adjust for these events, making it well-suited for business-related forecasting.

2) `Customizable Trend Components`: You can specify the components of the time series you want to model, such as trends, seasonality, and holidays. Prophet provides options for specifying custom seasonalities and holiday effects.

3) `Uncertainty Estimation`: Prophet provides uncertainty estimates for its forecasts, allowing you to understand the range of possible outcomes and the confidence intervals associated with the forecasts.

4) `Handling Missing Data`: Prophet can handle missing data points and outliers in a flexible manner, reducing the need for extensive data preprocessing.

5) `Scalability`: Prophet is designed to work well with large datasets, making it suitable for forecasting tasks involving a substantial amount of historical data.

6) `Interpretable Results`: The forecasts generated by Prophet are designed to be interpretable and explainable, making it easier to communicate forecasting results to stakeholders.

Prophet has gained popularity in industries that require accurate and intuitive time series forecasting, such as retail, e-commerce, finance, and more. It provides a higher-level interface compared to some other time series forecasting methods, making it accessible to users who might not have extensive experience in time series analysis. However, like any forecasting tool, it's important to evaluate Prophet's performance on your specific data and task to determine its suitability.


* Documentation: https://facebook.github.io/prophet/docs/quick_start.html#python-api

* Prophet AI Tutorial: https://www.kaggle.com/code/prashant111/tutorial-time-series-forecasting-with-prophet/notebook

![image](https://github.com/Ahmed-Sohail2000/Sun-Sense-Project/assets/107795296/88ebad8a-bc83-4218-a60b-1d67bb59d7c3)

The X-axis represents the years and the y-axis represents the solar irradiance values that depict the actual values which are indicated by black spots vs the predicted values which are the blue line. 

## 4.0 **Project Summary📓🧰**

![boy](https://github.com/Ahmed-Sohail2000/Sun-Sense-Project/assets/107795296/1200a53f-d833-4906-9b27-c5b6882395d2)

The `model 6 full data` compared against the `Prophet-AI` value for October 2nd, 2023 and it was measured against the actual solar energy value from 3 different solar pv systems. The results showed that prophet-ai predicted value of 5.67 kWh/m^2/day had an average energy deviation of 93 kWh whilst model 6 full data on the other hand with a predicted value of 3.97 kWh/m^/day had an average energy deviation of 218.3 kWh which means that `Prophet-AI` was much accurate compared to `model 6 full data`. The model can be further improved by implementing additional weather variables correlating to the solar irradiance values which would affect the solar energy production and produce a different result.

### Future Improvements 
There are some improvements and experiments that can be made in our time series analysis models and by applying and implementing this practice can definitely lead to understanding the concept of the proposed models and understand why one would perform this operation over the other.

1. **Hyperparameter Tuning:** Fine-tune the hyperparameters of your best-performing model. This could include adjusting learning rates, batch sizes, the number of layers or units in your neural network, etc. Hyperparameter optimization techniques like grid search or random search can help with this.

2. **Ensemble Methods:** Experiment with different ensemble methods, such as stacking, bagging, or boosting. Combining the predictions of multiple models can often improve overall performance.

3. **Feature Engineering:** Explore additional features or transformations of existing features that could provide valuable information to your model. Domain-specific knowledge can be especially useful here.

4. **LSTM Variants:** LSTM networks come in several variants like Bidirectional LSTM (Bi-LSTM), Attention mechanisms, or Gated Recurrent Units (GRUs). Experiment with these variants to see if they improve your model's performance.

5. **Transfer Learning:** Investigate if pre-trained models or embeddings can be leveraged for your time series data. For example, you can use a pre-trained model and fine-tune it on your specific task.

6. **Anomaly Detection:** Consider adding an anomaly detection component to your model. This can be valuable for identifying unusual patterns or outliers in your time series data.

7. **Data Augmentation:** Generate synthetic data or augment your existing data to increase the diversity of your training dataset. Techniques like time warping, amplitude scaling, or adding noise can be helpful.

8. **Regularization:** Implement regularization techniques like dropout, L1/L2 regularization, or weight decay to prevent overfitting, especially if you have a complex model.

9. **Advanced Forecasting Techniques:** Explore more advanced forecasting methods like Prophet, ARIMA, or Exponential Smoothing. These models have different assumptions and may capture certain patterns better.

10. **Cross-Validation:** Use cross-validation techniques to get a better estimate of your model's performance and ensure its generalizability.

11. **Interpretable Models:** Depending on your use case, you may need interpretable models. Experiment with simpler models like linear regression or decision trees to understand the underlying relationships in your data.

12. **Monitoring and Deployment:** Develop a system for monitoring your model in production. Consider how you'll handle concept drift or changes in the data distribution over time.

13. **Scaling:** Think about how your model will scale as your dataset grows. Can it handle larger volumes of data efficiently?

14. **Visualization:** Create informative visualizations of your time series data and model predictions. Visualizations can help in understanding and communicating results.

15. **Documentation:** Document your code, experiments, and results thoroughly. This will be invaluable for reproducibility and sharing your work with others.

16. **Feedback Loop:** If this project is part of a larger system or business process, establish a feedback loop to continuously improve your models as more data becomes available.

### Extra Curriculum & Resources

Extra Curriculum - https://dev.mrdbourke.com/tensorflow-deep-learning/10_time_series_forecasting_in_tensorflow/#extra-curriculum7

Resources

1. **Towards Data Science on Medium:** This publication offers a wide range of articles and tutorials on machine learning and data science. You can find it here: [Towards Data Science on Medium](https://towardsdatascience.com/).

2. **Kaggle:** Kaggle is a popular platform for data science and machine learning competitions. They have a wealth of notebooks and datasets to learn from, as well as a community forum for discussions. Check it out here: [Kaggle](https://www.kaggle.com/).

3. **GitHub:** Many machine learning practitioners share their code and projects on GitHub. You can search for repositories related to your specific interests and learn from the code and documentation. Start your search here: [GitHub](https://github.com/).

4. **Coursera and edX:** These platforms offer online courses in machine learning and data science. Some of them are provided by top universities and can be a great resource for in-depth learning. Coursera: [Coursera](https://www.coursera.org/), edX: [edX](https://www.edx.org/).

5. **Fast.ai:** Fast.ai provides free online courses and practical deep learning resources. They focus on making deep learning accessible to a broad audience. Explore their materials here: [Fast.ai](https://www.fast.ai/).

6. **YouTube:** Many machine learning experts and educators share tutorials and lectures on YouTube. Channels like "3Blue1Brown" and "sentdex" offer insightful content.

7. **Books:** Consider reading books like "Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow" by Aurélien Géron or "Deep Learning" by Ian Goodfellow, Yoshua Bengio, and Aaron Courville.









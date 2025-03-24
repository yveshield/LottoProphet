# -*- coding:utf-8 -*-
"""
Machine Learning Models for Lottery Prediction
Author: Yang Zhao

多种机器学习模型支持模块
包含XGBoost、随机森林等多种预测模型以及集成学习功能
"""
import os
import time
import torch
import numpy as np
import pandas as pd
import pickle
import joblib
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import logging
import json

# 尝试导入可选的模型库
try:
    from lightgbm import LGBMRegressor
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

# 定义支持的模型类型
MODEL_TYPES = {
    'random_forest': '随机森林',
    'xgboost': 'XGBoost',
    'gbdt': '梯度提升树',
    'ensemble': '集成模型'
}

# 如果可选库可用，添加到支持的模型中
if LIGHTGBM_AVAILABLE:
    MODEL_TYPES['lightgbm'] = 'LightGBM'
if CATBOOST_AVAILABLE:
    MODEL_TYPES['catboost'] = 'CatBoost'

class LotteryMLModels:
    """彩票预测机器学习模型类"""
    
    def __init__(self, lottery_type='dlt', model_type='ensemble', feature_window=10, log_callback=None, use_gpu=False):
        """
        初始化模型
        
        Args:
            lottery_type: 彩票类型，'dlt'或'ssq'
            model_type: 模型类型，可选值见MODEL_TYPES
            feature_window: 特征窗口大小，使用多少期数据作为特征
            log_callback: 日志回调函数，用于将日志发送到UI
            use_gpu: 是否使用GPU训练
        """
        self.lottery_type = lottery_type
        self.model_type = model_type
        self.feature_window = feature_window
        self.models = {}
        self.scalers = {}
        self.feature_cols = []
        self.log_callback = log_callback
        self.use_gpu = use_gpu
        
        # 添加将要保存的原始模型
        self.raw_models = {}
        
        # 初始化logger
        self.logger = logging.getLogger(f"ml_models_{lottery_type}")
        self.logger.setLevel(logging.INFO)
        # 防止重复添加handler
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
        # 记录GPU使用情况
        if self.use_gpu:
            self.log(f"ML模型使用GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else '不可用'}")
        else:
            self.log("ML模型使用CPU")
        
        # 选择基于彩票类型的球号范围
        if lottery_type == 'dlt':
            # 大乐透: 5个红球(1-35)，2个蓝球(1-12)
            self.red_range = 35
            self.blue_range = 12
            self.red_count = 5
            self.blue_count = 2
        else:  # ssq
            # 双色球: 6个红球(1-33)，1个蓝球(1-16)
            self.red_range = 33
            self.blue_range = 16
            self.red_count = 6
            self.blue_count = 1
        
        # 设置模型保存路径
        self.models_dir = os.path.join(f'./model/{lottery_type}')
        os.makedirs(self.models_dir, exist_ok=True)
    
    def log(self, message):
        """记录日志并发送到UI（如果有回调）"""
        self.logger.info(message)
        if self.log_callback:
            self.log_callback(message)
    
    def prepare_data(self, df, test_size=0.2):
        """准备训练数据"""
        # 提取数据
        self.log("准备训练数据...")
        
        # 特征窗口大小
        window_size = self.feature_window
        
        # 确保数据按期数排序
        df = df.sort_values('期数').reset_index(drop=True)
        
        # 提取红蓝球列名
        if self.lottery_type == 'dlt':
            red_cols = [col for col in df.columns if col.startswith('红球_')][:5]
            blue_cols = [col for col in df.columns if col.startswith('蓝球_')][:2]
        else:  # ssq
            red_cols = [col for col in df.columns if col.startswith('红球_')][:6]
            # 确保正确识别蓝球列，考虑两种可能的前缀格式
            blue_cols = []
            for col in df.columns:
                if col.startswith('蓝球_') or col == '蓝球':
                    blue_cols.append(col)
                    if len(blue_cols) >= 1:  # 双色球只有1个蓝球
                        break
        
        self.feature_cols = red_cols + blue_cols
        
        # 创建序列数据和标签
        X_data = []
        y_red = []
        y_blue = []
        
        for i in range(len(df) - window_size):
            # 创建特征序列
            features = []
            for j in range(window_size):
                row_features = []
                for col in red_cols + blue_cols:
                    row_features.append(df.iloc[i + j][col])
                features.append(row_features)
            
            # 添加特征
            X_data.append(features)
            
            # 添加标签
            red_labels = []
            for col in red_cols:
                red_labels.append(df.iloc[i + window_size][col] - 1)  # 减1使号码从0开始，适合分类模型
            y_red.append(red_labels)
            
            # 添加蓝球标签
            blue_labels = []
            for col in blue_cols:
                blue_labels.append(df.iloc[i + window_size][col] - 1)  # 减1使号码从0开始，适合分类模型
            y_blue.append(blue_labels)
        
        # 转换为NumPy数组
        X = np.array(X_data)
        y_red = np.array(y_red)
        y_blue = np.array(y_blue)
        
        # 记录数据形状
        self.log(f"特征形状: {X.shape}")
        self.log(f"红球标签形状: {y_red.shape}")
        self.log(f"蓝球标签形状: {y_blue.shape}")
        
        # 将3D特征展平为2D，以便用于传统ML模型
        X_reshaped = X.reshape(X.shape[0], -1)
        
        # 数据标准化，仅对输入特征进行处理，不对标签进行处理
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_reshaped)
        
        # 保存缩放器以便预测时使用
        self.scalers['X'] = scaler
        
        # 拆分训练集和测试集
        X_train, X_test, y_red_train, y_red_test, y_blue_train, y_blue_test = train_test_split(
            X_scaled, y_red, y_blue, test_size=test_size, random_state=42
        )
        
        # 对于ML模型，需要将多维标签展平为单维
        if y_red_train.ndim > 1:
            # 对于多个红球，随机选择一个位置进行预测
            red_pos = np.random.randint(0, y_red_train.shape[1])
            self.red_pos = red_pos
            y_red_class = y_red_train[:, red_pos]
        else:
            y_red_class = y_red_train
        
        # 对于蓝球标签，检查维度并确保安全访问
        if y_blue_train.shape[1] > 0:  # 确保数组有列可以访问
            # 对于多个蓝球，随机选择一个位置进行预测
            blue_pos = np.random.randint(0, y_blue_train.shape[1])
            self.blue_pos = blue_pos
            y_blue_class = y_blue_train[:, blue_pos]
        else:
            # 处理双色球可能没有蓝球的情况
            self.log("警告: 蓝球标签维度为0, 使用默认值0")
            y_blue_class = np.zeros(y_blue_train.shape[0], dtype=int)
            self.blue_pos = 0
        
        self.log(f"训练集特征形状: {X_train.shape}")
        
        return X_train, X_test, y_red_class, y_red_test[:, self.red_pos], y_blue_class, y_blue_test[:, self.blue_pos] if y_blue_test.shape[1] > 0 else np.zeros(y_blue_test.shape[0], dtype=int)
    
    def train_random_forest(self, X_train, y_train, ball_type, n_estimators=100):
        """训练随机森林模型"""
        from sklearn.ensemble import RandomForestClassifier
        
        self.log(f"训练{ball_type}球随机森林模型...")
        self.log(f"数据维度: 特征={X_train.shape}, 标签={y_train.shape}")
        self.log(f"随机森林参数: n_estimators={n_estimators}, max_depth=10")
        
        # 创建模型
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=10,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42,
            n_jobs=-1,
            verbose=1  # 启用详细输出
        )
        
        # 训练模型
        self.log(f"开始训练随机森林模型...")
        model.fit(X_train, y_train)
        
        # 输出特征重要性
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            self.log(f"随机森林特征重要性 (前10个):")
            max_features = min(10, len(indices))
            for i in range(max_features):
                feature_idx = indices[i]
                feature_name = f"特征_{feature_idx}" if feature_idx >= len(self.feature_cols) else self.feature_cols[feature_idx]
                self.log(f"  {i+1}. {feature_name}: {importances[feature_idx]:.4f}")
        
        self.log(f"{ball_type}球随机森林模型训练完成")
        return model
    
    # 添加处理多维预测的静态方法，使其可序列化
    @staticmethod
    def process_multidim_prediction(raw_preds):
        """处理多维预测结果，返回类别索引"""
        if len(raw_preds.shape) > 1 and raw_preds.shape[1] > 1:
            return np.argmax(raw_preds, axis=1)
        return raw_preds
    
    def train_xgboost(self, X_train, y_train, ball_type):
        """训练XGBoost模型"""
        import xgboost as xgb
        
        self.log(f"训练{ball_type}球XGBoost模型...")
        self.log(f"数据维度: 特征={X_train.shape}, 标签={y_train.shape}")
        
        # 设置参数，如果使用GPU，则使用GPU实现
        params = {
            'objective': 'multi:softmax',
            'num_class': self.red_range + 1 if ball_type == 'red' else self.blue_range + 1,
            'learning_rate': 0.1,
            'max_depth': 6,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'eval_metric': 'mlogloss',
            'verbosity': 0 
        }
        
        self.log(f"XGBoost参数: {params}")
        
        # 如果使用GPU并且GPU可用，则添加GPU参数
        if self.use_gpu and torch.cuda.is_available():
            params['tree_method'] = 'gpu_hist'  # 使用GPU加速
            self.log("XGBoost使用GPU加速训练")
        
        # 创建DMatrix数据结构
        dtrain = xgb.DMatrix(X_train, label=y_train)
        
        # 设置评估列表，用于记录训练进度
        watchlist = [(dtrain, 'train')]
        
        # 使用正确的XGBoost回调函数API
        class XGBCallback(xgb.callback.TrainingCallback):
            def __init__(self, log_func):
                self.log_func = log_func
                self.iteration = 0
                
            def after_iteration(self, model, epoch, evals_log):
                if (self.iteration + 1) % 10 == 0 or self.iteration == 0:  # 每10次迭代输出一次
                    # 从evals_log获取最新的评估结果
                    metric_values = evals_log.get('train', {}).get('mlogloss', [])
                    if metric_values:
                        msg = f'XGBoost迭代 {self.iteration + 1:3d}: {metric_values[-1]:.6f}'
                        self.log_func(msg)
                self.iteration += 1
                return False
        
        # 创建回调函数，确保始终使用回调
        callbacks = [XGBCallback(self.log)]
        
        # 训练模型
        self.log(f"开始训练XGBoost模型...")
        model = xgb.train(
            params, 
            dtrain, 
            num_boost_round=100, 
            evals=watchlist,
            callbacks=callbacks,
            verbose_eval=False  # 禁用内置的输出，使用我们的回调
        )
        
        # 输出特征重要性
        if model.feature_names:
            importance = model.get_score(importance_type='gain')
            sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
            
            self.log(f"XGBoost特征重要性 (前10个):")
            for i, (feature, score) in enumerate(sorted_importance[:10]):
                self.log(f"  {i+1}. {feature}: {score:.4f}")
        
        # 保存原始模型以便序列化
        self.raw_models[f'xgboost_{ball_type}'] = model
        
        # 使用类方法来包装模型预测，使其更容易序列化
        class WrappedXGBoostModel:
            def __init__(self, model, processor):
                self.model = model
                self.process_prediction = processor
                
            def predict(self, data):
                if not isinstance(data, xgb.DMatrix):
                    data = xgb.DMatrix(data)
                raw_preds = self.model.predict(data)
                return self.process_prediction(raw_preds)
        
        # 创建可序列化的包装模型
        wrapped_model = WrappedXGBoostModel(model, self.process_multidim_prediction)
        
        self.log(f"{ball_type}球XGBoost模型训练完成")
        return wrapped_model
    
    def train_gbdt(self, X_train, y_train, ball_type):
        """训练梯度提升决策树模型"""
        from sklearn.ensemble import GradientBoostingClassifier
        
        self.log(f"训练{ball_type}球GBDT模型...")
        self.log(f"数据维度: 特征={X_train.shape}, 标签={y_train.shape}")
        
        # 设置参数
        params = {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 5,
            'random_state': 42,
            'verbose': 0  # 减少自带的输出，使用我们自己的回调
        }
        
        self.log(f"GBDT参数: {params}")
        
        # 创建模型
        model = GradientBoostingClassifier(**params)
        
        # 自定义warm_start策略来监控训练进度
        self.log(f"开始训练GBDT模型...")
        
        # 使用分批训练方式来模拟进度报告
        n_estimators_per_batch = 10
        total_estimators = params['n_estimators']
        
        # 创建带有较小estimators的初始模型
        init_model = GradientBoostingClassifier(
            n_estimators=n_estimators_per_batch,
            learning_rate=params['learning_rate'],
            max_depth=params['max_depth'],
            random_state=params['random_state'],
            warm_start=True  # 允许增量训练
        )
        
        # 初始训练
        init_model.fit(X_train, y_train)
        
        # 逐步增加树的数量并继续训练
        for i in range(n_estimators_per_batch, total_estimators, n_estimators_per_batch):
            self.log(f"GBDT训练进度: {i}/{total_estimators} 棵树已完成 ({i/total_estimators*100:.1f}%)")
            init_model.n_estimators = min(i + n_estimators_per_batch, total_estimators)
            init_model.fit(X_train, y_train)
        
        self.log(f"GBDT训练进度: {total_estimators}/{total_estimators} 棵树已完成 (100%)")
        
        # 最终模型就是增量训练后的模型
        model = init_model
        
        # 输出特征重要性
        if hasattr(model, 'feature_importances_'):
            feature_names = [f"特征_{i}" for i in range(X_train.shape[1])]
            if len(self.feature_cols) == X_train.shape[1]:
                feature_names = self.feature_cols
                
            importance_data = sorted(zip(feature_names, model.feature_importances_), key=lambda x: x[1], reverse=True)
            
            self.log(f"GBDT特征重要性 (前10个):")
            for i, (feature, importance) in enumerate(importance_data[:10]):
                self.log(f"  {i+1}. {feature}: {importance:.4f}")
        
        # 保存原始模型以便序列化
        self.raw_models[f'gbdt_{ball_type}'] = model
        
        # 使用类方法来包装模型预测，使其更容易序列化
        class WrappedGBDTModel:
            def __init__(self, model, processor):
                self.model = model
                self.process_prediction = processor
                
            def predict(self, data):
                # 使用probabilities而不是类别
                raw_preds = model.predict_proba(data)
                return self.process_prediction(raw_preds)
        
        # 创建可序列化的包装模型
        wrapped_model = WrappedGBDTModel(model, self.process_multidim_prediction)
        
        self.log(f"{ball_type}球GBDT模型训练完成")
        return wrapped_model
    
    def train_lightgbm(self, X_train, y_train, ball_type):
        """训练LightGBM模型"""
        if not LIGHTGBM_AVAILABLE:
            self.log("LightGBM未安装，跳过此模型训练")
            return None
            
        import lightgbm as lgb
            
        self.log(f"训练{ball_type}球LightGBM模型...")
        self.log(f"数据维度: 特征={X_train.shape}, 标签={y_train.shape}")
        
        # 设置参数
        params = {
            'objective': 'multiclass',
            'num_class': self.red_range + 1 if ball_type == 'red' else self.blue_range + 1,
            'boosting_type': 'gbdt',
            'learning_rate': 0.1,
            'num_leaves': 31,
            'max_depth': -1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'metric': 'multi_logloss',
            'verbose': -1,  # 减少自带的输出，使用我们自己的回调
            'pred_early_stop': True,  # 早停预测
            'predict_disable_shape_check': True  # 禁用形状检查
        }
        
        self.log(f"LightGBM参数: {params}")
        
        # 如果使用GPU并且GPU可用，则添加GPU参数
        if self.use_gpu and torch.cuda.is_available():
            params['device'] = 'gpu'  # 使用GPU加速
            self.log("LightGBM使用GPU加速训练")
        
        # 创建数据集
        train_data = lgb.Dataset(X_train, label=y_train)
        
        # 设置评估列表，用于记录训练进度
        valid_sets = [train_data]
        valid_names = ['train']
        
        # 创建更可靠的回调函数以显示进度
        def progress_callback(env):
            # 确保每10次迭代或最后一次迭代时输出日志
            if env.iteration % 10 == 0 or env.iteration == env.end_iteration - 1:
                try:
                    # 获取评估结果
                    eval_result = env.evaluation_result_list
                    if eval_result and len(eval_result) > 0:
                        metric_name = eval_result[0][1]
                        metric_value = eval_result[0][2]
                        self.log(f"LightGBM迭代 {env.iteration+1}/{env.end_iteration}: {metric_name}={metric_value:.6f}")
                    else:
                        # 如果无法获取评估结果，至少显示进度
                        self.log(f"LightGBM迭代 {env.iteration+1}/{env.end_iteration}")
                except Exception as e:
                    # 出错时记录错误并继续
                    self.log(f"记录LightGBM进度时出错: {str(e)}")
            return False
        
        # 设置回调函数
        callbacks = [progress_callback]
        
        # 训练模型
        self.log(f"开始训练LightGBM模型...")
        model = lgb.train(
            params, 
            train_data, 
            num_boost_round=100,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks,
            verbose_eval=False  # 禁用内置的输出，使用我们的回调
        )
        
        # 输出特征重要性
        if hasattr(model, 'feature_importance'):
            try:
                importances = model.feature_importance(importance_type='gain')
                feature_names = model.feature_name()
                feature_importance = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
                
                self.log(f"LightGBM特征重要性 (前10个):")
                for i, (feature, importance) in enumerate(feature_importance[:10]):
                    self.log(f"  {i+1}. {feature}: {importance:.4f}")
            except Exception as e:
                self.log(f"获取LightGBM特征重要性时出错: {str(e)}")
        
        # 保存原始模型以便序列化
        self.raw_models[f'lightgbm_{ball_type}'] = model
        
        # 使用类方法来包装模型预测，使其更容易序列化
        class WrappedLightGBMModel:
            def __init__(self, model, processor):
                self.model = model
                self.process_prediction = processor
                
            def predict(self, data):
                raw_preds = self.model.predict(data)
                return self.process_prediction(raw_preds)
        
        # 创建可序列化的包装模型
        wrapped_model = WrappedLightGBMModel(model, self.process_multidim_prediction)
        
        self.log(f"{ball_type}球LightGBM模型训练完成")
        return wrapped_model
    
    def train_catboost(self, X_train, y_train, ball_type):
        """训练CatBoost模型"""
        if not CATBOOST_AVAILABLE:
            self.log("CatBoost未安装，跳过此模型训练")
            return None
            
        import catboost as cb
            
        self.log(f"训练{ball_type}球CatBoost模型...")
        self.log(f"数据维度: 特征={X_train.shape}, 标签={y_train.shape}")
        
        # 设置参数
        params = {
            'loss_function': 'MultiClass',
            'classes_count': self.red_range + 1 if ball_type == 'red' else self.blue_range + 1,
            'iterations': 100,
            'learning_rate': 0.1,
            'depth': 6,
            'random_seed': 42,
            'verbose': False,  # 减少自带的输出，使用我们自己的回调
            'train_dir': f'catboost_info_{ball_type}',  # 保存训练日志的目录
        }
        
        self.log(f"CatBoost参数: {params}")
        
        # 如果使用GPU并且GPU可用，则添加GPU参数
        if self.use_gpu and torch.cuda.is_available():
            params['task_type'] = 'GPU'  # 使用GPU加速
            self.log("CatBoost使用GPU加速训练")
        
        # 创建更可靠的自定义日志记录器，将输出发送到UI
        class LoggerCallback(cb.callback.Callback):
            def __init__(self, log_func):
                self.log_func = log_func
                self.last_info = {}
                
            def after_iteration(self, info):
                iter_num = info.iteration
                # 确保每10次迭代或最后一次迭代时输出日志
                if iter_num % 10 == 0 or iter_num == info.metric_info['iterations'] - 1:
                    try:
                        # 获取当前的评估指标
                        train_metrics = info.metrics.get('learn', {})
                        log_messages = []
                        
                        # 生成进度信息
                        progress_msg = f"CatBoost迭代 {iter_num+1}/{info.metric_info['iterations']}"
                        log_messages.append(progress_msg)
                        
                        # 添加指标信息
                        for metric_name, metric_values in train_metrics.items():
                            if len(metric_values) > 0:
                                value = metric_values[-1]
                                metric_msg = f"{metric_name}={value:.6f}"
                                log_messages.append(metric_msg)
                        
                        # 发送日志
                        if log_messages:
                            self.log_func(f"{': '.join(log_messages)}")
                        else:
                            # 如果没有指标信息，只显示进度
                            self.log_func(progress_msg)
                        
                    except Exception as e:
                        # 出错时记录错误并继续
                        self.log_func(f"记录CatBoost进度时出错: {str(e)}")
                        # 至少显示基本进度信息
                        self.log_func(f"CatBoost迭代 {iter_num+1}/{info.metric_info['iterations']}")
                        
                return True
        
        # 创建回调列表
        callbacks = [LoggerCallback(self.log)]
        
        # 训练模型
        self.log(f"开始训练CatBoost模型...")
        try:
            train_dataset = cb.Pool(X_train, y_train)
            model = cb.CatBoost(params)
            model.fit(train_dataset, callbacks=callbacks)
            
            # 输出特征重要性
            try:
                feature_importances = model.get_feature_importance()
                feature_names = [f"特征_{i}" for i in range(X_train.shape[1])]
                if len(self.feature_cols) == X_train.shape[1]:
                    feature_names = self.feature_cols
                    
                importance_data = sorted(zip(feature_names, feature_importances), key=lambda x: x[1], reverse=True)
                
                self.log(f"CatBoost特征重要性 (前10个):")
                for i, (feature, importance) in enumerate(importance_data[:10]):
                    self.log(f"  {i+1}. {feature}: {importance:.4f}")
            except Exception as e:
                self.log(f"获取CatBoost特征重要性时出错: {str(e)}")
            
            # 保存原始模型以便序列化
            self.raw_models[f'catboost_{ball_type}'] = model
            
            # 使用类方法来包装模型预测，使其更容易序列化
            class WrappedCatBoostModel:
                def __init__(self, model, processor):
                    self.model = model
                    self.process_prediction = processor
                    
                def predict(self, data):
                    raw_preds = self.model.predict(data)
                    return self.process_prediction(raw_preds)
            
            # 创建可序列化的包装模型
            wrapped_model = WrappedCatBoostModel(model, self.process_multidim_prediction)
            
            self.log(f"{ball_type}球CatBoost模型训练完成")
            return wrapped_model
            
        except Exception as e:
            self.log(f"训练CatBoost模型时出错: {str(e)}")
            import traceback
            self.log(traceback.format_exc())
            return None
    
    def train_ensemble(self, X_train, y_train, ball_type):
        """训练集成模型（包含多种基础模型）"""
        self.log(f"训练{ball_type}球集成模型...")
        self.log(f"数据维度: 特征={X_train.shape}, 标签={y_train.shape}")
        
        # 整合所有可用模型的预测结果
        ensemble_models = {}
        total_models = 5 if LIGHTGBM_AVAILABLE and CATBOOST_AVAILABLE else 3
        current_model = 0
        
        self.log(f"集成模型将训练以下子模型: 随机森林, XGBoost, GBDT{', LightGBM' if LIGHTGBM_AVAILABLE else ''}{', CatBoost' if CATBOOST_AVAILABLE else ''}")
        
        # 训练随机森林模型
        current_model += 1
        self.log(f"集成模型进度: 开始训练子模型 ({current_model}/{total_models}) - 随机森林")
        rf_model = self.train_random_forest(X_train, y_train, ball_type)
        if rf_model:
            ensemble_models['random_forest'] = rf_model
            self.log(f"集成模型进度: 随机森林模型添加完成 ({current_model}/{total_models})")
            
        # 训练XGBoost模型
        current_model += 1
        self.log(f"集成模型进度: 开始训练子模型 ({current_model}/{total_models}) - XGBoost")
        xgb_model = self.train_xgboost(X_train, y_train, ball_type)
        if xgb_model:
            ensemble_models['xgboost'] = xgb_model
            self.log(f"集成模型进度: XGBoost模型添加完成 ({current_model}/{total_models})")
            
        # 训练GBDT模型
        current_model += 1
        self.log(f"集成模型进度: 开始训练子模型 ({current_model}/{total_models}) - GBDT")
        gbdt_model = self.train_gbdt(X_train, y_train, ball_type)
        if gbdt_model:
            ensemble_models['gbdt'] = gbdt_model
            self.log(f"集成模型进度: GBDT模型添加完成 ({current_model}/{total_models})")
            
        # 训练LightGBM模型（如果可用）
        if LIGHTGBM_AVAILABLE:
            current_model += 1
            self.log(f"集成模型进度: 开始训练子模型 ({current_model}/{total_models}) - LightGBM")
            lgb_model = self.train_lightgbm(X_train, y_train, ball_type)
            if lgb_model:
                ensemble_models['lightgbm'] = lgb_model
                self.log(f"集成模型进度: LightGBM模型添加完成 ({current_model}/{total_models})")
                
        # 训练CatBoost模型（如果可用）
        if CATBOOST_AVAILABLE:
            current_model += 1
            self.log(f"集成模型进度: 开始训练子模型 ({current_model}/{total_models}) - CatBoost")
            cb_model = self.train_catboost(X_train, y_train, ball_type)
            if cb_model:
                ensemble_models['catboost'] = cb_model
                self.log(f"集成模型进度: CatBoost模型添加完成 ({current_model}/{total_models})")
        
        self.log(f"集成模型完成，包含 {len(ensemble_models)} 个子模型")
        
        # 记录每个子模型的信息
        model_info = []
        for model_name in ensemble_models.keys():
            model_info.append(f"- {model_name}")
        
        self.log(f"集成模型包含以下子模型:\n" + "\n".join(model_info))
        
        return ensemble_models
    
    def train(self, df):
        """训练模型"""
        self.log("============ 开始训练模型 ============")
        self.log(f"彩票类型: {self.lottery_type.upper()}")
        self.log(f"选择的模型类型: {self.model_type}")
        
        training_start_time = time.time()
        
        # 准备数据
        self.log("\n====== 第1阶段: 数据准备 ======")
        data_prep_start = time.time()
        X_train, X_test, y_red_train, y_red_test, y_blue_train, y_blue_test = self.prepare_data(df)
        data_prep_time = time.time() - data_prep_start
        self.log(f"数据准备完成，耗时: {data_prep_time:.2f}秒")
        
        # 初始化模型字典
        self.models = {}
        
        # 根据选定的模型类型训练模型
        self.log("\n====== 第2阶段: 模型训练 ======")
        model_train_start = time.time()
        
        if self.model_type == 'random_forest':
            self.log("\n----- 使用随机森林模型 -----")
            self.models['red'] = self.train_random_forest(X_train, y_red_train, 'red')
            self.models['blue'] = self.train_random_forest(X_train, y_blue_train, 'blue')
        
        elif self.model_type == 'xgboost':
            self.log("\n----- 使用XGBoost模型 -----")
            self.models['red'] = self.train_xgboost(X_train, y_red_train, 'red')
            self.models['blue'] = self.train_xgboost(X_train, y_blue_train, 'blue')
        
        elif self.model_type == 'gbdt':
            self.log("\n----- 使用GBDT模型 -----")
            self.models['red'] = self.train_gbdt(X_train, y_red_train, 'red')
            self.models['blue'] = self.train_gbdt(X_train, y_blue_train, 'blue')
        
        elif self.model_type == 'lightgbm' and LIGHTGBM_AVAILABLE:
            self.log("\n----- 使用LightGBM模型 -----")
            self.models['red'] = self.train_lightgbm(X_train, y_red_train, 'red')
            self.models['blue'] = self.train_lightgbm(X_train, y_blue_train, 'blue')
        
        elif self.model_type == 'catboost' and CATBOOST_AVAILABLE:
            self.log("\n----- 使用CatBoost模型 -----")
            self.models['red'] = self.train_catboost(X_train, y_red_train, 'red')
            self.models['blue'] = self.train_catboost(X_train, y_blue_train, 'blue')
        
        elif self.model_type == 'ensemble':
            self.log("\n----- 使用集成模型 -----")
            self.models['red'] = self.train_ensemble(X_train, y_red_train, 'red')
            self.models['blue'] = self.train_ensemble(X_train, y_blue_train, 'blue')
        
        model_train_time = time.time() - model_train_start
        self.log(f"\n模型训练完成，耗时: {model_train_time:.2f}秒")
        
        # 评估模型
        self.log("\n====== 第3阶段: 模型评估 ======")
        eval_start = time.time()
        red_acc, blue_acc = self.evaluate(X_test, y_red_test, y_blue_test)
        eval_time = time.time() - eval_start
        self.log(f"模型评估完成，耗时: {eval_time:.2f}秒")
        
        # 保存模型
        self.log("\n====== 第4阶段: 模型保存 ======")
        save_start = time.time()
        self.save_models()
        save_time = time.time() - save_start
        self.log(f"模型保存完成，耗时: {save_time:.2f}秒")
        
        total_time = time.time() - training_start_time
        self.log("\n============ 训练总结 ============")
        self.log(f"总训练时间: {total_time:.2f}秒")
        self.log(f"数据准备: {data_prep_time:.2f}秒 ({data_prep_time/total_time*100:.1f}%)")
        self.log(f"模型训练: {model_train_time:.2f}秒 ({model_train_time/total_time*100:.1f}%)")
        self.log(f"模型评估: {eval_time:.2f}秒 ({eval_time/total_time*100:.1f}%)")
        self.log(f"模型保存: {save_time:.2f}秒 ({save_time/total_time*100:.1f}%)")
        self.log(f"红球预测准确率: {red_acc:.4f}")
        self.log(f"蓝球预测准确率: {blue_acc:.4f}")
        self.log(f"整体预测准确率: {(red_acc + blue_acc) / 2:.4f}")
        self.log("============ 训练结束 ============")
        
        return self.models
    
    def evaluate(self, X_test, y_red_test, y_blue_test):
        """评估模型性能"""
        self.log("评估模型性能...")
        
        # 处理1D测试标签
        if len(y_red_test.shape) == 1:
            y_red_test = y_red_test.reshape(-1, 1)
        if len(y_blue_test.shape) == 1:
            y_blue_test = y_blue_test.reshape(-1, 1)
            
        # 对每个球进行预测和评估
        red_accuracy = 0
        blue_accuracy = 0
        
        # 评估红球预测
        if 'red' in self.models:
            y_pred_red = self.models['red'].predict(X_test)
            
            # 检查预测结果的维度，适用于CatBoost模型等返回多维预测结果的情况
            if len(y_pred_red.shape) > 1 and y_pred_red.shape[1] > 1:
                # 如果是多分类输出，取每行的最大值索引作为预测类别
                self.log(f"处理多维预测结果，形状: {y_pred_red.shape}")
                y_pred_red = np.argmax(y_pred_red, axis=1)
            
            # 确保形状匹配
            y_red_test_flat = y_red_test.flatten()
            y_pred_red_flat = y_pred_red.flatten()
            
            # 记录预测和实际值的形状以便调试
            self.log(f"红球预测形状: {y_pred_red_flat.shape}, 真实值形状: {y_red_test_flat.shape}")
            
            # 计算准确率
            red_accuracy = np.mean(y_pred_red_flat == y_red_test_flat)
            self.log(f"红球预测准确率: {red_accuracy:.4f}")
        
        # 评估蓝球预测
        if 'blue' in self.models:
            y_pred_blue = self.models['blue'].predict(X_test)
            
            # 检查预测结果的维度，适用于CatBoost模型等返回多维预测结果的情况
            if len(y_pred_blue.shape) > 1 and y_pred_blue.shape[1] > 1:
                # 如果是多分类输出，取每行的最大值索引作为预测类别
                self.log(f"处理多维预测结果，形状: {y_pred_blue.shape}")
                y_pred_blue = np.argmax(y_pred_blue, axis=1)
            
            # 确保形状匹配
            y_blue_test_flat = y_blue_test.flatten()
            y_pred_blue_flat = y_pred_blue.flatten()
            
            # 记录预测和实际值的形状以便调试
            self.log(f"蓝球预测形状: {y_pred_blue_flat.shape}, 真实值形状: {y_blue_test_flat.shape}")
            
            # 计算准确率
            blue_accuracy = np.mean(y_pred_blue_flat == y_blue_test_flat)
            self.log(f"蓝球预测准确率: {blue_accuracy:.4f}")
        
        # 计算整体准确率
        overall_accuracy = (red_accuracy + blue_accuracy) / 2
        self.log(f"整体预测准确率: {overall_accuracy:.4f}")
        
        return red_accuracy, blue_accuracy
    
    def save_models(self):
        """保存模型和缩放器"""
        model_dir = os.path.join(self.models_dir, self.model_type)
        os.makedirs(model_dir, exist_ok=True)
        
        # 保存模型信息文件，记录模型类型和创建时间
        model_info = {
            'model_type': self.model_type,
            'lottery_type': self.lottery_type,
            'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
            'feature_window': self.feature_window,
            'red_count': self.red_count,
            'blue_count': self.blue_count
        }
        info_path = os.path.join(model_dir, 'model_info.json')
        with open(info_path, 'w') as f:
            json.dump(model_info, f, indent=4)
        
        self.log(f"保存模型信息到 {info_path}")
        
        # 分别保存不同类型的原始模型
        if self.model_type == 'lightgbm' and LIGHTGBM_AVAILABLE:
            try:
                import lightgbm as lgb
                # 使用LightGBM的内置保存方法
                if 'lightgbm_red' in self.raw_models:
                    red_model_path = os.path.join(model_dir, 'red_model.txt')
                    self.raw_models['lightgbm_red'].save_model(red_model_path)
                if 'lightgbm_blue' in self.raw_models:
                    blue_model_path = os.path.join(model_dir, 'blue_model.txt')
                    self.raw_models['lightgbm_blue'].save_model(blue_model_path)
                self.log(f"LightGBM模型已保存到 {model_dir}")
            except Exception as e:
                self.log(f"保存LightGBM模型出错: {str(e)}")
        
        elif self.model_type == 'xgboost':
            try:
                # 使用XGBoost的内置保存方法
                if 'xgboost_red' in self.raw_models:
                    red_model_path = os.path.join(model_dir, 'red_model.json')
                    self.raw_models['xgboost_red'].save_model(red_model_path)
                if 'xgboost_blue' in self.raw_models:
                    blue_model_path = os.path.join(model_dir, 'blue_model.json')
                    self.raw_models['xgboost_blue'].save_model(blue_model_path)
                self.log(f"XGBoost模型已保存到 {model_dir}")
            except Exception as e:
                self.log(f"保存XGBoost模型出错: {str(e)}")
        
        elif self.model_type == 'catboost' and CATBOOST_AVAILABLE:
            try:
                # 使用CatBoost的内置保存方法
                if 'catboost_red' in self.raw_models:
                    red_model_path = os.path.join(model_dir, 'red_model.cbm')
                    self.raw_models['catboost_red'].save_model(red_model_path)
                if 'catboost_blue' in self.raw_models:
                    blue_model_path = os.path.join(model_dir, 'blue_model.cbm')
                    self.raw_models['catboost_blue'].save_model(blue_model_path)
                self.log(f"CatBoost模型已保存到 {model_dir}")
            except Exception as e:
                self.log(f"保存CatBoost模型出错: {str(e)}")
        
        else:
            # 对于其他模型类型，使用joblib保存
            try:
                model_path = os.path.join(model_dir, 'models.pkl')
                joblib.dump(self.models, model_path)
                self.log(f"模型已保存到 {model_path}")
            except Exception as e:
                self.log(f"保存模型出错: {str(e)}")
        
        # 保存缩放器
        scaler_path = os.path.join(model_dir, 'scalers.pkl')
        joblib.dump(self.scalers, scaler_path)
        
        # 保存特征列名
        feature_path = os.path.join(model_dir, 'features.pkl')
        joblib.dump(self.feature_cols, feature_path)
        
        self.log(f"模型相关数据已保存到 {model_dir}")
    
    def load_models(self):
        """加载模型和缩放器"""
        model_dir = os.path.join(self.models_dir, self.model_type)
        
        if not os.path.exists(model_dir):
            self.log(f"模型目录不存在: {model_dir}")
            return False
        
        try:
            # 加载模型信息
            info_path = os.path.join(model_dir, 'model_info.json')
            if os.path.exists(info_path):
                with open(info_path, 'r') as f:
                    model_info = json.load(f)
                    self.log(f"加载模型信息: {model_info}")
            
            # 根据模型类型加载不同的模型
            if self.model_type == 'lightgbm' and LIGHTGBM_AVAILABLE:
                import lightgbm as lgb
                self.models = {}
                self.raw_models = {}
                
                # 加载红球模型
                red_model_path = os.path.join(model_dir, 'red_model.txt')
                if os.path.exists(red_model_path):
                    lgb_model = lgb.Booster(model_file=red_model_path)
                    self.raw_models['lightgbm_red'] = lgb_model
                    
                    # 创建包装模型
                    class WrappedLightGBMModel:
                        def __init__(self, model, processor):
                            self.model = model
                            self.process_prediction = processor
                            
                        def predict(self, data):
                            raw_preds = self.model.predict(data)
                            return self.process_prediction(raw_preds)
                    
                    self.models['red'] = WrappedLightGBMModel(lgb_model, self.process_multidim_prediction)
                
                # 加载蓝球模型
                blue_model_path = os.path.join(model_dir, 'blue_model.txt')
                if os.path.exists(blue_model_path):
                    lgb_model = lgb.Booster(model_file=blue_model_path)
                    self.raw_models['lightgbm_blue'] = lgb_model
                    self.models['blue'] = WrappedLightGBMModel(lgb_model, self.process_multidim_prediction)
                
                self.log(f"LightGBM模型已从 {model_dir} 加载")
            
            elif self.model_type == 'xgboost':
                import xgboost as xgb
                self.models = {}
                self.raw_models = {}
                
                # 加载红球模型
                red_model_path = os.path.join(model_dir, 'red_model.json')
                if os.path.exists(red_model_path):
                    xgb_model = xgb.Booster()
                    xgb_model.load_model(red_model_path)
                    self.raw_models['xgboost_red'] = xgb_model
                    
                    # 创建包装模型
                    class WrappedXGBoostModel:
                        def __init__(self, model, processor):
                            self.model = model
                            self.process_prediction = processor
                            
                        def predict(self, data):
                            if not isinstance(data, xgb.DMatrix):
                                data = xgb.DMatrix(data)
                            raw_preds = self.model.predict(data)
                            return self.process_prediction(raw_preds)
                    
                    self.models['red'] = WrappedXGBoostModel(xgb_model, self.process_multidim_prediction)
                
                # 加载蓝球模型
                blue_model_path = os.path.join(model_dir, 'blue_model.json')
                if os.path.exists(blue_model_path):
                    xgb_model = xgb.Booster()
                    xgb_model.load_model(blue_model_path)
                    self.raw_models['xgboost_blue'] = xgb_model
                    self.models['blue'] = WrappedXGBoostModel(xgb_model, self.process_multidim_prediction)
                
                self.log(f"XGBoost模型已从 {model_dir} 加载")
            
            elif self.model_type == 'catboost' and CATBOOST_AVAILABLE:
                import catboost as cb
                self.models = {}
                self.raw_models = {}
                
                # 加载红球模型
                red_model_path = os.path.join(model_dir, 'red_model.cbm')
                if os.path.exists(red_model_path):
                    cb_model = cb.CatBoost()
                    cb_model.load_model(red_model_path)
                    self.raw_models['catboost_red'] = cb_model
                    
                    # 创建包装模型
                    class WrappedCatBoostModel:
                        def __init__(self, model, processor):
                            self.model = model
                            self.process_prediction = processor
                            
                        def predict(self, data):
                            raw_preds = self.model.predict(data)
                            return self.process_prediction(raw_preds)
                    
                    self.models['red'] = WrappedCatBoostModel(cb_model, self.process_multidim_prediction)
                
                # 加载蓝球模型
                blue_model_path = os.path.join(model_dir, 'blue_model.cbm')
                if os.path.exists(blue_model_path):
                    cb_model = cb.CatBoost()
                    cb_model.load_model(blue_model_path)
                    self.raw_models['catboost_blue'] = cb_model
                    self.models['blue'] = WrappedCatBoostModel(cb_model, self.process_multidim_prediction)
                
                self.log(f"CatBoost模型已从 {model_dir} 加载")
            
            else:
                # 加载其他类型的模型
                model_path = os.path.join(model_dir, 'models.pkl')
                self.models = joblib.load(model_path)
                
                # 加载缩放器
                scaler_path = os.path.join(model_dir, 'scalers.pkl')
                if os.path.exists(scaler_path):
                    self.scalers = joblib.load(scaler_path)
                
                # 加载特征列名
                feature_path = os.path.join(model_dir, 'features.pkl')
                if os.path.exists(feature_path):
                    self.feature_cols = joblib.load(feature_path)
                
                self.log(f"模型已从 {model_dir} 加载")
                return True
        except Exception as e:
            self.log(f"加载模型时出错: {str(e)}")
            import traceback
            self.log(traceback.format_exc())
            return False
    
    def prepare_prediction_data(self, recent_data):
        """
        准备预测数据
        
        Args:
            recent_data: 包含最近开奖数据的DataFrame
            
        Returns:
            预处理后的特征数据
        """
        # 定义红蓝球列名
        if self.lottery_type == 'dlt':
            red_cols = [col for col in recent_data.columns if col.startswith('红球_')][:5]
            blue_cols = [col for col in recent_data.columns if col.startswith('蓝球_')][:2]
        else:  # ssq
            red_cols = [col for col in recent_data.columns if col.startswith('红球_')][:6]
            blue_cols = [col for col in recent_data.columns if col.startswith('蓝球')][:1]
        
        # 按期数降序排列
        recent_data = recent_data.sort_values('期数', ascending=False).reset_index(drop=True)
        
        # 确保有足够的历史数据
        if len(recent_data) < self.feature_window:
            self.log(f"历史数据不足，需要至少 {self.feature_window} 期")
            return None
        
        # 创建特征
        features = []
        for i in range(self.feature_window):
            features.extend(recent_data.iloc[i][red_cols].values)
            features.extend(recent_data.iloc[i][blue_cols].values)
            
        X = np.array([features])
        
        # 使用缩放器进行标准化
        if 'X' in self.scalers:
            X_scaled = self.scalers['X'].transform(X)
            return X_scaled
        else:
            self.log("缩放器未找到，无法预处理数据")
            return None
    
    def predict(self, recent_data):
        """
        生成预测结果
        
        Args:
            recent_data: 包含最近开奖数据的DataFrame
            
        Returns:
            预测的红球和蓝球号码
        """
        # 提取红蓝球列名
        if self.lottery_type == 'dlt':
            red_cols = [col for col in recent_data.columns if col.startswith('红球_')][:5]
            blue_cols = [col for col in recent_data.columns if col.startswith('蓝球_')][:2]
        else:  # ssq
            red_cols = [col for col in recent_data.columns if col.startswith('红球_')][:6]
            blue_cols = [col for col in recent_data.columns if col.startswith('蓝球_')][:1]
            
        # 确保数据按期数降序排列
        recent_data = recent_data.sort_values('期数', ascending=False).reset_index(drop=True)
        
        # 确保有足够的历史数据
        if len(recent_data) < self.feature_window:
            self.log(f"历史数据不足，需要至少 {self.feature_window} 期")
            return None, None
        
        # 创建特征序列
        X_data = []
        
        # 使用滑动窗口创建序列数据
        features = []
        for j in range(self.feature_window):
            row_features = []
            for col in red_cols + blue_cols:
                row_features.append(recent_data.iloc[j][col])
            features.append(row_features)
            
        X_data.append(features)
        
        # 转换为NumPy数组
        X = np.array(X_data)
        
        # 重塑特征以适合模型
        X = X.reshape(X.shape[0], -1)
        
        # 确保模型已加载
        if 'red' not in self.models or 'blue' not in self.models:
            self.log("模型未加载，无法预测")
            return None, None
        
        # 预测红球
        red_pred = self.models['red'].predict(X)[0]
        if self.model_type == 'ensemble':
            # 集成模型需要单独处理
            red_votes = {}
            for name, model in self.models['red'].items():
                preds = model.predict(X)[0]
                for pred in preds:
                    if pred not in red_votes:
                        red_votes[pred] = 0
                    red_votes[pred] += 1
            red_predictions = sorted(red_votes.items(), key=lambda x: x[1], reverse=True)[:self.red_count]
            red_predictions = [p[0] + 1 for p in red_predictions]  # +1 转回原始号码范围
        else:
            # 单一模型预测
            red_predictions = [int(red_pred) + 1]  # +1 转回原始号码范围
        
        # 预测蓝球
        blue_pred = self.models['blue'].predict(X)[0]
        if self.model_type == 'ensemble':
            # 集成模型需要单独处理
            blue_votes = {}
            for name, model in self.models['blue'].items():
                preds = model.predict(X)[0]
                for pred in preds:
                    if pred not in blue_votes:
                        blue_votes[pred] = 0
                    blue_votes[pred] += 1
            blue_predictions = sorted(blue_votes.items(), key=lambda x: x[1], reverse=True)[:self.blue_count]
            blue_predictions = [p[0] + 1 for p in blue_predictions]  # +1 转回原始号码范围
        else:
            # 单一模型预测
            blue_predictions = [int(blue_pred) + 1]  # +1 转回原始号码范围
            
        # 生成完整号码集
        while len(red_predictions) < self.red_count:
            # 如果预测的红球数量不足，随机补充
            new_num = np.random.randint(1, self.red_range + 1)
            if new_num not in red_predictions:
                red_predictions.append(new_num)
                
        while len(blue_predictions) < self.blue_count:
            # 如果预测的蓝球数量不足，随机补充
            new_num = np.random.randint(1, self.blue_range + 1)
            if new_num not in blue_predictions:
                blue_predictions.append(new_num)
                
        # 确保号码不重复且按升序排列
        red_predictions = sorted(list(set(red_predictions)))[:self.red_count]
        blue_predictions = sorted(list(set(blue_predictions)))[:self.blue_count]
        
        return red_predictions, blue_predictions

# 使用示例
def demo():
    # 加载数据
    from scripts.data_analysis import load_lottery_data
    lottery_type = 'dlt'  # 或 'ssq'
    df = load_lottery_data(lottery_type)
    
    # 初始化模型
    model = LotteryMLModels(lottery_type=lottery_type, model_type='ensemble')
    
    # 训练模型
    model.train(df)
    
    # 预测下一期号码
    recent_data = df.sort_values('期数', ascending=False).head(10)
    red_predictions, blue_predictions = model.predict(recent_data)
    
    print(f"预测红球号码: {red_predictions}")
    print(f"预测蓝球号码: {blue_predictions}")

if __name__ == "__main__":
    demo() 
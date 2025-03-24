# -*- coding:utf-8 -*-
"""
UI Components for Lottery Predictor Application
Author: Yang Zhao
"""
from PyQt5.QtWidgets import (
    QVBoxLayout, QPushButton, QLabel, QComboBox, QWidget, 
    QTextEdit, QSpinBox, QHBoxLayout, QTabWidget, QScrollArea, 
    QGridLayout, QCheckBox, QGroupBox, QFormLayout, QMainWindow,
    QMenu, QAction
)
from PyQt5.QtCore import Qt
from theme_manager import ThemeManager, CustomThemeDialog
import torch
from model_utils import name_path

def create_main_tab(main_tab):
    """
    创建主标签页的UI组件
    
    Args:
        main_tab: QWidget，主标签页容器
        
    Returns:
        tuple: 主标签页中的关键UI组件
    """
    main_layout = QVBoxLayout(main_tab)
    main_layout.setSpacing(6)    
    main_layout.setContentsMargins(8, 8, 8, 8)  
    

    cuda_available = False
    cuda_device = "不可用"
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            cuda_device = torch.cuda.get_device_name(0)
    except:
        pass
    

    top_widget = QWidget()
    top_layout = QGridLayout()
    

    lottery_label = QLabel("彩票类型:")
    lottery_combo = QComboBox()
    lottery_combo.addItems([name_path[key]['name'] for key in name_path.keys()])
    
    model_label = QLabel("模型类型:")
    model_combo = QComboBox()

    model_combo.addItem("LSTM-CRF (默认)")

    from ml_models import MODEL_TYPES
    for model_key, model_name in MODEL_TYPES.items():
        model_combo.addItem(model_name)
    
    top_layout.addWidget(lottery_label, 0, 0)
    top_layout.addWidget(lottery_combo, 0, 1)
    top_layout.addWidget(model_label, 0, 2)
    top_layout.addWidget(model_combo, 0, 3)
    
    prediction_label = QLabel("预测数量:")
    prediction_spin = QSpinBox()
    prediction_spin.setRange(1, 10)
    prediction_spin.setValue(5)
    
    gpu_checkbox = QCheckBox("使用GPU训练")
    gpu_checkbox.setChecked(cuda_available)
    gpu_checkbox.setEnabled(cuda_available)
    if not cuda_available:
        gpu_checkbox.setToolTip("您的系统未安装GPU版本的PyTorch或没有可用的CUDA设备")
    else:
        gpu_checkbox.setToolTip(f"使用GPU加速训练 ({cuda_device})")
    
    predict_button = QPushButton("生成预测")
    
    top_layout.addWidget(prediction_label, 1, 0)
    top_layout.addWidget(prediction_spin, 1, 1)
    top_layout.addWidget(gpu_checkbox, 1, 2)
    top_layout.addWidget(predict_button, 1, 3)
    
    main_layout.addWidget(top_widget)
    
  
    theme_layout = QHBoxLayout()
    theme_label = QLabel("主题样式:")
    theme_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
    
    theme_combo = QComboBox()
    theme_combo.addItems(ThemeManager().get_theme_names())
    theme_combo.setCurrentText(ThemeManager().current_theme)
    
    customize_theme_button = QPushButton("自定义主题")
    
    theme_layout.addStretch()
    theme_layout.addWidget(theme_label)
    theme_layout.addWidget(theme_combo)
    theme_layout.addWidget(customize_theme_button)
    
    main_layout.addLayout(theme_layout)
    
  
    control_layout = QHBoxLayout()
    control_layout.setSpacing(10)
    
    settings_group = QGroupBox("预测设置")
    settings_layout = QFormLayout(settings_group)
    settings_layout.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
    settings_layout.setSpacing(6)
    
    settings_layout.addRow("彩票类型:", lottery_combo)
    settings_layout.addRow("预测数量:", prediction_spin)
    settings_layout.addRow("预测模型:", model_combo)

    gpu_group = QGroupBox("GPU设置")
    gpu_layout = QVBoxLayout(gpu_group)
    gpu_layout.setSpacing(6)
    
    gpu_layout.addWidget(gpu_checkbox)
    
    control_layout.addWidget(settings_group, 2)
    control_layout.addWidget(gpu_group, 1)
    
    main_layout.addLayout(control_layout)
    
   
    button_layout = QHBoxLayout()
    button_layout.setSpacing(6)
    
    train_button = QPushButton("训练模型")
    train_button.setMinimumHeight(30)
    
    pause_button = QPushButton("暂停训练")
    pause_button.setMinimumHeight(30)
    pause_button.setEnabled(False)
    
    analyze_button = QPushButton("数据分析")
    analyze_button.setMinimumHeight(30)
    
    update_data_button = QPushButton("更新数据")
    update_data_button.setMinimumHeight(30)
    
    button_layout.addWidget(predict_button)
    button_layout.addWidget(train_button)
    button_layout.addWidget(pause_button)
    button_layout.addWidget(analyze_button)
    button_layout.addWidget(update_data_button)
    
    main_layout.addLayout(button_layout)

    
    content_layout = QHBoxLayout()
    
    
    result_group = QGroupBox("预测结果")
    result_layout = QVBoxLayout(result_group)
    
    result_label = QLabel("点击'生成预测'按钮查看预测结果")
    result_label.setAlignment(Qt.AlignCenter)
    result_label.setWordWrap(True)
    result_label.setStyleSheet("padding: 10px; background-color: white; border: 1px solid #DDDDDD;")
    result_label.setMinimumHeight(200)
    
    result_layout.addWidget(result_label)
    
    
    log_group = QGroupBox("训练和预测日志")
    log_layout = QVBoxLayout(log_group)
    
    log_box = QTextEdit()
    log_box.setReadOnly(True)
    log_box.setStyleSheet("font-family: Consolas, monospace; font-size: 10pt;")
    
    # 启用右键菜单
    log_box.setContextMenuPolicy(Qt.CustomContextMenu)
    
    log_layout.addWidget(log_box)
    
    content_layout.addWidget(result_group, 1)
    content_layout.addWidget(log_group, 2)
    
    main_layout.addLayout(content_layout, 1)
    
    return (predict_button, train_button, pause_button, analyze_button, update_data_button,
            lottery_combo, prediction_spin, gpu_checkbox, result_label, log_box,
            theme_combo, customize_theme_button, model_combo)


def create_analysis_tab(analysis_tab):
    """
    创建数据分析标签页的UI组件
    
    Args:
        analysis_tab: QWidget，数据分析标签页容器
        
    Returns:
        tuple: 数据分析标签页中的关键UI组件
    """
    analysis_layout = QVBoxLayout(analysis_tab)
    analysis_layout.setSpacing(6)
    analysis_layout.setContentsMargins(8, 8, 8, 8)
    

    analysis_control_layout = QHBoxLayout()
    
    chart_group = QGroupBox("图表选择")
    chart_layout = QHBoxLayout(chart_group)
    
    chart_layout.addWidget(QLabel("分析图表:"))
    analysis_combo = QComboBox()
    analysis_combo.addItems(["频率分布", "热冷号码", "遗漏分析", "号码模式", "趋势分析"])
    analysis_combo.setMinimumWidth(150)
    chart_layout.addWidget(analysis_combo)
    
    trend_feature_combo = QComboBox()
    trend_feature_combo.setVisible(False)  
    trend_feature_combo.setMinimumWidth(150)
    chart_layout.addWidget(trend_feature_combo)
    chart_layout.addStretch()
    
    analysis_control_layout.addWidget(chart_group)
    analysis_layout.addLayout(analysis_control_layout)
    
   
    analysis_content = QHBoxLayout()
    
    chart_group = QGroupBox("数据可视化")
    chart_layout = QVBoxLayout(chart_group)
    
    chart_label = QLabel("请先点击'数据分析'按钮加载数据")
    chart_label.setAlignment(Qt.AlignCenter)
    chart_label.setMinimumHeight(350)
    chart_label.setStyleSheet("border: 1px dashed #CCCCCC; background-color: white;")
    
    chart_layout.addWidget(chart_label)
    
    stats_group = QGroupBox("统计信息摘要")
    stats_layout = QVBoxLayout(stats_group)
    
    stats_text = QTextEdit()
    stats_text.setReadOnly(True)
    stats_text.setMaximumHeight(150)
    stats_text.setStyleSheet("font-family: Consolas, monospace; font-size: 10pt;")
    
    stats_layout.addWidget(stats_text)
    
    analysis_content.addWidget(chart_group, 7)
    analysis_content.addWidget(stats_group, 3)
    
    analysis_layout.addLayout(analysis_content, 1)
    
    # 添加新的统计分析按钮
    advanced_stats_button = QPushButton("高级统计分析")
    analysis_layout.addWidget(advanced_stats_button)
    
    distribution_analysis_button = QPushButton("分布分析")
    analysis_layout.addWidget(distribution_analysis_button)
    
    return (analysis_combo, trend_feature_combo, chart_label, stats_text,
            advanced_stats_button, distribution_analysis_button)


def create_advanced_statistics_tab(advanced_stats_tab):
    """
    创建高级统计分析标签页的UI组件
    
    Args:
        advanced_stats_tab: QWidget，高级统计分析标签页容器
        
    Returns:
        tuple: 高级统计分析标签页中的关键UI组件
    """
    advanced_stats_layout = QVBoxLayout(advanced_stats_tab)
    
  
    settings_group = QGroupBox("设置")
    settings_layout = QHBoxLayout()
    

    lottery_type_label = QLabel("彩票类型:")
    advanced_stats_lottery_combo = QComboBox()
    advanced_stats_lottery_combo.addItems(["双色球", "大乐透"])
    settings_layout.addWidget(lottery_type_label)
    settings_layout.addWidget(advanced_stats_lottery_combo)
    

    run_advanced_stats_button = QPushButton("运行高级统计分析")
    settings_layout.addWidget(run_advanced_stats_button)
    
    run_distribution_analysis_button = QPushButton("运行分布分析")
    settings_layout.addWidget(run_distribution_analysis_button)
    
    show_stats_data_button = QPushButton("显示统计数据")
    settings_layout.addWidget(show_stats_data_button)
    
   
    settings_group.setLayout(settings_layout)
    advanced_stats_layout.addWidget(settings_group)
    
 
    results_group = QGroupBox("分析结果")
    results_layout = QVBoxLayout()
    
  
    scroll_area = QScrollArea()
    scroll_area.setWidgetResizable(True)
    scroll_content = QWidget()
    scroll_layout = QVBoxLayout(scroll_content)
    

    stats_result_label = QLabel("请选择彩票类型并运行分析...")
    stats_result_label.setAlignment(Qt.AlignCenter)
    scroll_layout.addWidget(stats_result_label)

    scroll_area.setWidget(scroll_content)
    results_layout.addWidget(scroll_area)

    results_group.setLayout(results_layout)
    advanced_stats_layout.addWidget(results_group)
    
    return (advanced_stats_lottery_combo, run_advanced_stats_button,
            run_distribution_analysis_button, show_stats_data_button,
            stats_result_label)


def create_main_window():
    """
    创建主窗口实例
    
    Returns:
        LotteryPredictorApp: 主窗口实例
    """
    from lottery_predictor_app_new import LotteryPredictorApp
    return LotteryPredictorApp() 
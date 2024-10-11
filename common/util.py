# -*- coding: utf-8 -*-

import logging
import os


def setup_logger(log_file=None, level=logging.INFO):
    """设置日志系统"""
    logger = logging.getLogger()

    # 检查是否已经有处理器添加到 logger 中，避免重复添加
    if not logger.hasHandlers():
        # 设置日志格式
        formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(message)s')

        # 设置控制台输出
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # 如果指定了日志文件，设置文件输出
        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        # 设置日志级别
        logger.setLevel(level)

    return logger

# knowledge_base.py
import yaml
import os
import re
from typing import Dict, Any, Optional, List

class PhoneKnowledgeBase:
    """手机评测知识库"""

    def __init__(self, yaml_path: str = None):
        """初始化知识库，自动查找 YAML 文件"""
        if yaml_path is None:
            yaml_path = self._find_yaml()
        self.yaml_path = yaml_path
        self.knowledge = self._load_knowledge()

    def _find_yaml(self) -> str:
        """从当前目录向上查找 phone_knowledge.yaml"""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        possible_paths = [
            os.path.join(current_dir, "phone_knowledge.yaml"),
            os.path.join(os.path.dirname(current_dir), "phone_knowledge.yaml"),
            os.path.join(os.getcwd(), "phone_knowledge.yaml"),
            os.path.join(os.path.dirname(os.getcwd()), "phone_knowledge.yaml"),
        ]
        for path in possible_paths:
            if os.path.exists(path):
                return path
        # 默认返回当前目录下的文件名（后续加载会报错）
        return os.path.join(current_dir, "phone_knowledge.yaml")

    def _load_knowledge(self) -> Dict:
        """加载知识库文件"""
        try:
            with open(self.yaml_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            print(f"⚠️ 知识库文件不存在: {self.yaml_path}")
            return {}
        except Exception as e:
            print(f"⚠️ 加载知识库失败: {e}")
            return {}

    # ==================== 芯片查询（改进匹配逻辑） ====================
    def get_chipset_info(self, chipset_name: str) -> Optional[Dict]:
        """获取芯片信息，支持精确匹配和智能模糊匹配"""
        if not chipset_name:
            return None

        chipsets = self.knowledge.get("chipsets", {})
        if not chipsets:
            return None

        # 1. 精确匹配
        if chipset_name in chipsets:
            return chipsets[chipset_name]

        # 2. 前缀匹配：键以输入开头，且输入后第一个非空白字符不是字母数字
        input_len = len(chipset_name)
        for key, value in chipsets.items():
            if key.startswith(chipset_name):
                rest = key[input_len:].lstrip()
                if not rest or not rest[0].isalnum():
                    return value

        # 3. 回退：包含匹配，并选择键最短的（最接近原型号）
        best_match = None
        best_key_len = float('inf')
        for key, value in chipsets.items():
            if chipset_name in key or key in chipset_name:
                if len(key) < best_key_len:
                    best_key_len = len(key)
                    best_match = value
        return best_match

    # ==================== 屏幕指南 ====================
    def get_display_guide(self, resolution: str = "", refresh_rate: str = "",
                          screen_type: str = "", brightness: str = "") -> Dict:
        """获取屏幕指南"""
        displays = self.knowledge.get("displays", {})
        result = {}

        # 分辨率指南
        if resolution:
            resolution_guide = displays.get("resolution_guide", {})
            for key, desc in resolution_guide.items():
                if key.lower() in resolution.lower():
                    result["resolution_desc"] = desc
                    break

        # 刷新率指南
        if refresh_rate:
            refresh_guide = displays.get("refresh_rate_guide", {})
            rate_num = self._extract_number(refresh_rate)
            if rate_num:
                for key, desc in refresh_guide.items():
                    key_num = self._extract_number(key)
                    if key_num and abs(rate_num - key_num) <= 15:
                        result["refresh_desc"] = desc
                        break

        # 屏幕材质指南
        if screen_type:
            screen_type_guide = displays.get("screen_type_guide", {})
            for key, desc in screen_type_guide.items():
                if key.lower() in screen_type.lower():
                    result["screen_type_desc"] = desc
                    break

        # 亮度指南
        if brightness:
            brightness_guide = displays.get("brightness_guide", {})
            brightness_num = self._extract_number(brightness)
            if brightness_num:
                for key, desc in brightness_guide.items():
                    if "以上" in key and brightness_num >= self._extract_number(key):
                        result["brightness_desc"] = desc
                        break
                    elif "以下" in key and brightness_num <= self._extract_number(key):
                        result["brightness_desc"] = desc
                        break
                    elif "-" in key:
                        nums = [self._extract_number(n) for n in key.split("-")]
                        if len(nums) == 2 and nums[0] <= brightness_num <= nums[1]:
                            result["brightness_desc"] = desc
                            break

        return result

    # ==================== 电池指南 ====================
    def get_battery_guide(self, capacity: str = "", charging: str = "",
                          wireless_charging: str = "") -> Dict:
        """获取电池指南"""
        battery = self.knowledge.get("battery", {})
        result = {}

        # 电池容量指南
        if capacity:
            capacity_guide = battery.get("capacity_guide", {})
            cap_num = self._extract_number(capacity)
            if cap_num:
                for key, desc in capacity_guide.items():
                    if "以上" in key and cap_num >= self._extract_number(key):
                        result["capacity_desc"] = desc
                        break
                    elif "以下" in key and cap_num <= self._extract_number(key):
                        result["capacity_desc"] = desc
                        break
                    elif "-" in key:
                        nums = [self._extract_number(n) for n in key.split("-")]
                        if len(nums) == 2 and nums[0] <= cap_num <= nums[1]:
                            result["capacity_desc"] = desc
                            break

        # 有线充电指南
        if charging:
            charging_guide = battery.get("charging_guide", {})
            charge_num = self._extract_number(charging)
            if charge_num:
                for key, desc in charging_guide.items():
                    if "以上" in key and charge_num >= self._extract_number(key):
                        result["charging_desc"] = desc
                        break
                    elif "以下" in key and charge_num <= self._extract_number(key):
                        result["charging_desc"] = desc
                        break
                    elif "-" in key:
                        nums = [self._extract_number(n) for n in key.split("-")]
                        if len(nums) == 2 and nums[0] <= charge_num <= nums[1]:
                            result["charging_desc"] = desc
                            break

        # 无线充电指南
        if wireless_charging:
            wireless_guide = battery.get("wireless_charging_guide", {})
            wireless_num = self._extract_number(wireless_charging)
            if wireless_num:
                for key, desc in wireless_guide.items():
                    if "以上" in key and wireless_num >= self._extract_number(key):
                        result["wireless_charging_desc"] = desc
                        break
                    elif "以下" in key and wireless_num <= self._extract_number(key):
                        result["wireless_charging_desc"] = desc
                        break
                    elif "-" in key:
                        nums = [self._extract_number(n) for n in key.split("-")]
                        if len(nums) == 2 and nums[0] <= wireless_num <= nums[1]:
                            result["wireless_charging_desc"] = desc
                            break

        return result

    # ==================== 相机指南 ====================
    def get_camera_guide(self, sensor_size: str = "", aperture: str = "",
                         resolution: str = "", zoom: str = "") -> Dict:
        """获取相机指南"""
        camera = self.knowledge.get("camera", {})
        result = {}

        # 传感器尺寸指南
        if sensor_size:
            sensor_guide = camera.get("sensor_guide", {})
            for key, desc in sensor_guide.items():
                if key in sensor_size:
                    result["sensor_desc"] = desc
                    break

        # 光圈指南
        if aperture:
            aperture_guide = camera.get("aperture_guide", {})
            aperture_num = self._extract_float(aperture)
            if aperture_num:
                for key, desc in aperture_guide.items():
                    if "-" in key:
                        nums = [self._extract_float(n) for n in key.split("-")]
                        if len(nums) == 2 and nums[0] <= aperture_num <= nums[1]:
                            result["aperture_desc"] = desc
                            break
                    elif "以上" in key and aperture_num >= self._extract_float(key):
                        result["aperture_desc"] = desc
                        break

        # 像素指南
        if resolution:
            pixel_guide = camera.get("pixel_guide", {})
            pixel_num = self._extract_number(resolution)
            if pixel_num:
                for key, desc in pixel_guide.items():
                    key_num = self._extract_number(key)
                    if key_num and abs(pixel_num - key_num) <= 1000:
                        result["pixel_desc"] = desc
                        break

        # 变焦指南
        if zoom:
            zoom_guide = camera.get("zoom_guide", {})
            zoom_num = self._extract_float(zoom)
            if zoom_num:
                for key, desc in zoom_guide.items():
                    if "以上" in key and zoom_num >= self._extract_float(key):
                        result["zoom_desc"] = desc
                        break
                    elif "以下" in key and zoom_num <= self._extract_float(key):
                        result["zoom_desc"] = desc
                        break
                    elif "-" in key:
                        nums = [self._extract_float(n) for n in key.split("-")]
                        if len(nums) == 2 and nums[0] <= zoom_num <= nums[1]:
                            result["zoom_desc"] = desc
                            break
                    else:
                        # 无区间标记的键（如"20倍光学品质变焦"）按数值相等匹配
                        key_num = self._extract_float(key)
                        if key_num is not None and abs(zoom_num - key_num) < 0.1:
                            result["zoom_desc"] = desc
                            break

        return result

    # ==================== 品牌信息 ====================
    def get_brand_info(self, brand: str) -> Optional[Dict]:
        """获取品牌信息"""
        if not brand:
            return None

        brands = self.knowledge.get("brands", {})

        # 精确匹配
        if brand in brands:
            return brands[brand]

        # 模糊匹配（不区分大小写）
        brand_lower = brand.lower()
        for key, value in brands.items():
            if key.lower() in brand_lower or brand_lower in key.lower():
                return value
        return None

    # ==================== 传感器信息 ====================
    def get_sensor_info(self, sensor_name: str) -> Optional[Dict]:
        """获取传感器详细信息"""
        if not sensor_name:
            return None

        sensors = self.knowledge.get("sensors", {})

        # 精确匹配
        if sensor_name in sensors:
            return sensors[sensor_name]

        # 模糊匹配
        sensor_lower = sensor_name.lower()
        for key, value in sensors.items():
            if key.lower() in sensor_lower or sensor_lower in key.lower():
                return value
        return None

    # ==================== 场景推荐 ====================
    def get_scenario_recommendations(self, scenario: str) -> Optional[Dict]:
        """获取使用场景建议"""
        scenarios = self.knowledge.get("usage_scenarios", {})
        return scenarios.get(scenario)

    # ==================== 知识总结生成 ====================
    def generate_knowledge_summary(self, params: Dict) -> Dict:
        """根据提取的参数生成知识总结"""
        summary = {}

        # 处理器
        processor = params.get("processor", {})
        soc = processor.get("soc", "")
        if soc:
            chipset_info = self.get_chipset_info(soc)
            if chipset_info:
                summary["chipset"] = {
                    "name": soc,
                    "info": chipset_info
                }

        # 屏幕
        display = params.get("display", {})
        if display:
            display_guide = self.get_display_guide(
                resolution=display.get("resolution", ""),
                refresh_rate=display.get("refresh_rate", ""),
                screen_type=display.get("type", ""),
                brightness=display.get("peak_brightness", "")
            )
            if display_guide:
                summary["display"] = display_guide

        # 电池
        battery = params.get("battery_charging", {})
        if battery:
            battery_guide = self.get_battery_guide(
                capacity=battery.get("capacity", ""),
                charging=battery.get("wired_charging", ""),
                wireless_charging=battery.get("wireless_charging", "")
            )
            if battery_guide:
                summary["battery"] = battery_guide

        # 相机
        camera = params.get("camera", {})
        rear_main = camera.get("rear_main", {})
        if rear_main:
            camera_guide = self.get_camera_guide(
                sensor_size=rear_main.get("sensor_size", ""),
                aperture=rear_main.get("aperture", ""),
                resolution=rear_main.get("resolution", ""),
                zoom=rear_main.get("optical_zoom", "")
            )
            if camera_guide:
                summary["camera"] = camera_guide

        # 品牌
        basic_info = params.get("basic_info", {})
        brand = basic_info.get("brand", "")
        if brand:
            brand_info = self.get_brand_info(brand)
            if brand_info:
                summary["brand"] = brand_info

        return summary

    # ==================== 辅助函数 ====================
    def _extract_number(self, text: str) -> Optional[float]:
        """从文本中提取第一个数字"""
        if not text:
            return None
        try:
            numbers = re.findall(r'\d+\.?\d*', text)
            if numbers:
                return float(numbers[0])
        except:
            pass
        return None

    def _extract_float(self, text: str) -> Optional[float]:
        """提取浮点数（同 _extract_number）"""
        return self._extract_number(text)
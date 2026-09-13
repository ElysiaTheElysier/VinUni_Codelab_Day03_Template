"""
Lab #3: Baseline Chatbot vs ReAct Agent
Học viên hoàn thiện các mục TODO để hoàn thành bài lab.
"""

import json
import os
import sys
from typing import Dict, Any, Tuple
from tools import TOOL_DEFINITIONS, TOOL_MAP, get_flight_info, get_weather_forecast

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

SYSTEM_PROMPT = """Bạn là một ReAct Agent thông minh hỗ trợ khách hàng Vingroup.
Bạn chỉ sử dụng các công cụ sau:
{tools}

Quy trình trả lời bắt buộc:
Thought: <Suy nghĩ bước tiếp theo>
Action: {{"name": "<tên tool>", "args": {{<tham số>}}}}
Observation: <Kết quả từ tool>
... (Lặp lại cho tới khi có đủ dữ liệu)
Final Answer: <Câu trả lời hoàn chỉnh cho khách hàng>
"""

class ChatbotBaseline:
    """Baseline LLM Chatbot without ReAct Loop or Tools"""
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")

    def query(self, user_input: str) -> Dict[str, Any]:
        if self.api_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                for model_name in ['gemini-3.6-flash', 'gemini-3.8-flash', 'gemini-2.5-flash', 'gemini-1.5-flash']:
                    try:
                        model = genai.GenerativeModel(model_name)
                        response = model.generate_content(
                            f"Bạn là chatbot tư vấn du lịch. Hãy trả lời câu hỏi sau một cách tự nhiên KHÔNG dùng tool hay internet: {user_input}"
                        )
                        return {
                            "answer": response.text,
                            "tool_calls": [],
                            "status": "success",
                            "mode": "live_api"
                        }
                    except Exception:
                        continue
            except Exception:
                pass

        user_lower = user_input.lower()
        if any(g in user_lower for g in ["hi", "hello", "chào", "bạn là ai"]):
            answer = "Xin chào! Tôi là Chatbot tư vấn du lịch cơ bản (Baseline). Tôi có thể trò chuyện chung với bạn, nhưng tôi không có quyền truy cập cơ sở dữ liệu thời gian thực để tra cứu vé máy bay hay thời tiết chính xác."
        elif any(k in user_lower for k in ["chuyến bay", "vé"]):
            answer = "Bạn có thể tìm chuyến bay trên các trang hàng không. Tôi không thể tra cứu giá vé và chuyến bay theo thời gian thực."
        elif any(k in user_lower for k in ["thời tiết", "nhiệt độ", "mặc gì"]):
            answer = "Về thời tiết, bạn nên tra cứu trên trang dự báo thời tiết uy tín. Tôi không có dữ liệu khí tượng trực tiếp."
        else:
            answer = "Bạn có thể tìm chuyến bay trên các trang hàng không. Về thời tiết, bạn nên tra cứu trên trang dự báo thời tiết."

        return {
            "answer": answer,
            "tool_calls": [],
            "status": "success",
            "mode": "mock_baseline"
        }

class ReActAgent:
    """ReAct Agent có sử dụng Thought-Action-Observation Loop"""
    def __init__(self, max_iterations: int = 5):
        self.max_iterations = max_iterations
        self.trace = []

    def parse_city_code(self, text: str) -> str:
        text_upper = text.upper()
        for code in ["SGN", "HAN", "DAD"]:
            if code in text_upper:
                return code
        if "HÀ NỘI" in text_upper:
            return "HAN"
        if "HỒ CHÍ MINH" in text_upper or "SÀI GÒN" in text_upper:
            return "SGN"
        if "ĐÀ NẮNG" in text_upper:
            return "DAD"
        return "SGN"

    def plan_and_execute_step(self, user_input: str, iteration: int) -> Tuple[str, bool]:
        """Lập kế hoạch và thực thi từng bước trong vòng lặp ReAct"""
        user_lower = user_input.lower()
        
        # 1. Câu hỏi FAQ chung: Không cần gọi tool
        if "chính sách" in user_lower or "đổi trả" in user_lower:
            thought = "Đây là câu hỏi FAQ chung về chính sách. Không cần sử dụng tool."
            final_answer = "Vé máy bay Vinpearl có thể hỗ trợ đổi ngày trước 24 giờ so với giờ khởi hành, phí đổi vé là 350.000 VNĐ/vé cộng chênh lệch giá vé (nếu có)."
            self.trace.append({"iteration": iteration, "thought": thought, "final_answer": final_answer})
            return final_answer, True

        # 2. Câu hỏi chào hỏi / Smalltalk: Không cần gọi tool
        greetings = ["hi", "hello", "chào", "bạn là ai", "alo", "helo", "hey"]
        if any(g == user_lower.strip() or g in user_lower.split() for g in greetings) and not any(k in user_lower for k in ["bay", "vé", "thời tiết", "trang phục"]):
            thought = "Người dùng đang chào hỏi hoặc bắt đầu hội thoại. Tôi sẽ gửi lời chào và giới thiệu năng lực hỗ trợ du lịch mà không dùng tool."
            final_answer = "Xin chào! Tôi là Trợ lý AI Du lịch thông minh của Vingroup. Tôi có thể hỗ trợ bạn:\n- ✈️ Tra cứu chuyến bay nội địa (Hà Nội, TP.HCM, Đà Nẵng) theo ngân sách.\n- ☀️ Xem dự báo thời tiết và gợi ý trang phục tại điểm đến.\n- 📋 Giải đáp chính sách vé máy bay Vinpearl.\n\nBạn cần tôi hỗ trợ thông tin gì hôm nay?"
            self.trace.append({"iteration": iteration, "thought": thought, "final_answer": final_answer})
            return final_answer, True

        # Xác định các nhu cầu cần gọi tool
        needs_flight = any(k in user_lower for k in ["chuyến bay", "vé", "bay từ", "vé máy bay"])
        needs_weather = any(k in user_lower for k in ["thời tiết", "mặc gì", "nhiệt độ", "mưa"])

        # 3. Nếu không liên quan đến chuyến bay hay thời tiết (ngoài phạm vi)
        if not needs_flight and not needs_weather and iteration == 1:
            thought = "Yêu cầu của người dùng nằm ngoài phạm vi tra cứu chuyến bay và thời tiết của tôi."
            final_answer = "Tôi là Trợ lý Du lịch chuyên trách về chuyến bay và thời tiết của Vingroup. Hiện tại tôi chỉ có dữ liệu tra cứu các chặng bay nội địa (HAN, SGN, DAD) và thông tin thời tiết điểm đến. Bạn vui lòng đặt câu hỏi liên quan đến chuyến bay hoặc thời tiết nhé!"
            self.trace.append({"iteration": iteration, "thought": thought, "final_answer": final_answer})
            return final_answer, True

        # 2. Xử lý tra cứu chuyến bay ở iteration 1
        if needs_flight and iteration == 1:
            origin = "HAN" if "han" in user_lower or "hà nội" in user_lower else "DAD"
            destination = "DAD" if "dad" in user_lower or "đà nẵng" in user_lower else "SGN"
            max_price = 5000000
            if "2 triệu" in user_lower or "2.000.000" in user_lower:
                max_price = 2000000
            elif "1.5 triệu" in user_lower or "1,5 triệu" in user_lower:
                max_price = 1500000
            elif "500k" in user_lower:
                max_price = 500000

            thought = f"Tôi cần tra cứu chuyến bay từ {origin} đi {destination} với giá tối đa {max_price} VND."
            action = {"name": "get_flight_info", "args": {"origin": origin, "destination": destination, "max_price": max_price}}
            obs = TOOL_MAP["get_flight_info"](**action["args"])
            
            self.trace.append({
                "iteration": iteration,
                "thought": thought,
                "action": action,
                "observation": obs
            })
            
            # Nếu người dùng chỉ hỏi vé bay (không hỏi thời tiết) -> Hoàn tất luôn ở bước 1
            if not needs_weather:
                if not obs:
                    final_ans = f"Không tìm thấy chuyến bay nào từ {origin} đi {destination} dưới {max_price:,} VND."
                else:
                    lines = [f"- {fl['airline']} ({fl['flight_number']}): {fl['departure_time']} - Giá: {fl['price_vnd']:,} VNĐ" for fl in obs]
                    final_ans = f"Tìm thấy {len(obs)} chuyến bay từ {origin} đi {destination}:\n" + "\n".join(lines)
                return final_ans, True
                
            return f"Thought: {thought}\nAction: {json.dumps(action, ensure_ascii=False)}\nObservation: {json.dumps(obs, ensure_ascii=False)}", False

        # 3. Xử lý tra cứu thời tiết
        elif needs_weather and (iteration == 2 or (iteration == 1 and not needs_flight)):
            city_code = self.parse_city_code(user_input)
            thought = f"Tôi cần kiểm tra thông tin thời tiết tại {city_code}."
            action = {"name": "get_weather_forecast", "args": {"city_code": city_code}}
            obs = TOOL_MAP["get_weather_forecast"](**action["args"])

            self.trace.append({
                "iteration": iteration,
                "thought": thought,
                "action": action,
                "observation": obs
            })

            # Nếu người dùng chỉ hỏi thời tiết (không hỏi vé bay) -> Hoàn tất luôn
            if not needs_flight:
                final_ans = f"Thời tiết tại {obs.get('city', city_code)}: {obs.get('temperature_c', 'N/A')}°C, {obs.get('condition', '')}.\nGợi ý: {obs.get('recommendation', '')}"
                return final_ans, True
                
            return f"Thought: {thought}\nAction: {json.dumps(action, ensure_ascii=False)}\nObservation: {json.dumps(obs, ensure_ascii=False)}", False

        # 4. Bước tổng hợp Final Answer (khi đã thu thập đủ thông tin cả vé bay và thời tiết)
        else:
            thought = "Tôi đã thu thập đủ thông tin để trả lời khách hàng."
            flight_obs = next((t["observation"] for t in self.trace if t.get("action", {}).get("name") == "get_flight_info"), [])
            weather_obs = next((t["observation"] for t in self.trace if t.get("action", {}).get("name") == "get_weather_forecast"), {})

            flight_summary = "Không tìm thấy chuyến bay phù hợp."
            if flight_obs:
                lines = [f"   - {fl['airline']} ({fl['flight_number']}): {fl['departure_time']} - Giá: {fl['price_vnd']:,} VNĐ" for fl in flight_obs]
                flight_summary = "\n".join(lines)

            weather_summary = f"Thời tiết tại {weather_obs.get('city', 'địa phương')}: {weather_obs.get('temperature_c', '')}°C ({weather_obs.get('condition', '')}).\n   - Gợi ý trang phục: {weather_obs.get('recommendation', '')}"

            final_answer = (
                f"1. Thông tin chuyến bay:\n{flight_summary}\n\n"
                f"2. Thông tin thời tiết & trang phục:\n   - {weather_summary}"
            )
            self.trace.append({
                "iteration": iteration,
                "thought": thought,
                "final_answer": final_answer
            })
            return final_answer, True

    def run(self, user_input: str) -> Dict[str, Any]:
        self.trace = []
        iteration = 1
        
        # Vòng lặp ReAct với cơ chế Safeguard giới hạn số bước lặp
        while iteration <= self.max_iterations:
            result, is_final = self.plan_and_execute_step(user_input, iteration)
            if is_final:
                return {
                    "answer": result,
                    "trace": self.trace,
                    "iterations": iteration,
                    "status": "completed"
                }
            iteration += 1

        # Safeguard kích hoạt khi vượt quá max_iterations
        return {
            "answer": "Lỗi: Agent đã vượt quá số bước lặp tối đa (Max Iterations Safeguard).",
            "trace": self.trace,
            "iterations": iteration - 1,
            "status": "max_iterations_reached"
        }

def main():
    user_query = "Tìm cho tôi chuyến bay từ HAN đi SGN dưới 2 triệu, rồi cho biết thời tiết SGN nên mặc gì?"
    
    print("=== RUNNING CHATBOT BASELINE ===")
    chatbot = ChatbotBaseline()
    print(chatbot.query(user_query))
    
    print("\n=== RUNNING REACT AGENT ===")
    agent = ReActAgent(max_iterations=5)
    result = agent.run(user_query)
    print("Result:", result)
    print("Trace Log:", json.dumps(agent.trace, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
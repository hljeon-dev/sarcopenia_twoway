import json

class Config:
    def __init__(self, elbow_min, elbow_max, shoulder_min, shoulder_max, elbow_gradient, shoulder_gradient, 
                 knee_min, knee_max, hip_min, hip_max, knee_gradient, hip_gradient, out_path, speed, number, important_angle):
        self.elbow_min = elbow_min
        self.elbow_max = elbow_max
        self.shoulder_min = shoulder_min
        self.shoulder_max = shoulder_max
        self.elbow_gradient = elbow_gradient
        self.shoulder_gradient = shoulder_gradient
        self.knee_min = knee_min
        self.knee_max = knee_max
        self.hip_min = hip_min
        self.hip_max = hip_max
        self.knee_gradient = knee_gradient
        self.hip_gradient = hip_gradient
        self.out_path = out_path
        self.speed = speed
        self.number = number
        self.important_angle = important_angle

    @classmethod
    def from_json(cls, json_path, video_name):
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        config_data = data[video_name]   # video_name이 키 값이 되어 나머지 값들을 지정
        return cls(**config_data)   # Config 클래스의 인스턴스를 일일이 생성할 필요 없이 전체 생성(cls는 현재 클래스 참조, **config_data는 key argument)
    
    def to_dict(self):
        return {
            'elbow_min': self.elbow_min,
            'elbow_max': self.elbow_max,
            'shoulder_min': self.shoulder_min,
            'shoulder_max': self.shoulder_max,
            'elbow_gradient': self.elbow_gradient,
            'shoulder_gradient': self.shoulder_gradient,
            'knee_min': self.knee_min,
            'knee_max': self.knee_max,
            'hip_min': self.hip_min,
            'hip_max': self.hip_max,
            'knee_gradient': self.knee_gradient,
            'hip_gradient': self.hip_gradient,
            'out_path': self.out_path,
            'speed': self.speed,
            'number': self.number,
            'important_angle': self.important_angle
        }
import cv2 as cv
import mediapipe as mp
import numpy as np
from collections import deque, Counter
import time
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# Modern color scheme
BACKGROUND_COLOR = (25, 25, 25)
TEXT_COLOR = (240, 240, 240)
ACCENT_COLOR_1 = (0, 150, 255)  # Orange
ACCENT_COLOR_2 = (0, 255, 150)  # Cyan
NEUTRAL_COLOR = (150, 150, 150)

class GameState:
    def __init__(self):
        self.clock = 0
        self.p1_move = self.p2_move = None
        self.gameText = ""
        self.success = True 
        self.p1_score = self.p2_score = 0
        self.round_history = deque(maxlen=50)
        self.round_active = False
        self.wait_for_reset = False
        self.game_mode = "Normal"  # Normal, Best of 3, Best of 5
        self.rounds_played = 0
        self.start_time = time.time()

game_state = GameState()

def getHandMove(hand_landmarks):
    landmarks = hand_landmarks.landmark
    
    # Improved gesture recognition
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    middle_tip = landmarks[12]
    ring_tip = landmarks[16]
    pinky_tip = landmarks[20]
    wrist = landmarks[0]
    
    # Check if fingers are extended
    fingers_extended = [
        thumb_tip.x < index_tip.x,  # Thumb
        index_tip.y < landmarks[6].y,  # Index
        middle_tip.y < landmarks[10].y,  # Middle
        ring_tip.y < landmarks[14].y,  # Ring
        pinky_tip.y < landmarks[18].y  # Pinky
    ]
    
    if sum(fingers_extended) <= 1:
        return "rock"
    elif sum(fingers_extended) == 5:
        return "paper"
    elif fingers_extended[1] and fingers_extended[2] and not fingers_extended[3] and not fingers_extended[4]:
        return "scissors"
    else:
        return "unknown"

def draw_text(frame, text, position, font, scale, color, thickness):
    cv.putText(frame, text, position, font, scale, color, thickness, cv.LINE_AA)

def draw_modern_text(frame, text, position, font, scale, color, thickness):
    shadow_offset = 2
    draw_text(frame, text, (position[0] + shadow_offset, position[1] + shadow_offset), font, scale, BACKGROUND_COLOR, thickness + 1)
    draw_text(frame, text, position, font, scale, color, thickness)

def create_overlay(width, height):
    overlay = np.zeros((height, width, 3), dtype=np.uint8)
    cv.rectangle(overlay, (0, 0), (width, height), BACKGROUND_COLOR, -1)
    return overlay

def update_game_state(results):
    global game_state
    
    if not game_state.round_active and not game_state.wait_for_reset:
        if 0 <= game_state.clock < 20:
            game_state.success = True
            game_state.gameText = "Ready?"
        elif game_state.clock < 30:
            game_state.gameText = "3.."
        elif game_state.clock < 40:
            game_state.gameText = "2.."
        elif game_state.clock < 50:
            game_state.gameText = "1.."
        elif game_state.clock < 60:
            game_state.gameText = "GO!"
        elif game_state.clock == 60:
            game_state.round_active = True
            hls = results.multi_hand_landmarks
            if hls and len(hls) == 2:
                game_state.p1_move = getHandMove(hls[0])
                game_state.p2_move = getHandMove(hls[1])
            else:
                game_state.success = False
    elif game_state.round_active:
        if game_state.success:
            game_state.gameText = f"Player 1: {game_state.p1_move}. Player 2: {game_state.p2_move}."
            if game_state.p1_move == game_state.p2_move:
                result = "Tie"
            elif (game_state.p1_move == "paper" and game_state.p2_move == "rock") or \
                 (game_state.p1_move == "rock" and game_state.p2_move == "scissors") or \
                 (game_state.p1_move == "scissors" and game_state.p2_move == "paper"):
                result = "Player 1 wins"
                game_state.p1_score += 1
            else:
                result = "Player 2 wins"
                game_state.p2_score += 1
            game_state.gameText += f" {result}!"
            game_state.round_history.appendleft((game_state.p1_move, game_state.p2_move, result))
            game_state.round_active = False
            game_state.wait_for_reset = True
            game_state.rounds_played += 1
        else:
            game_state.gameText = "Didn't play properly!"
            game_state.round_active = False
            game_state.wait_for_reset = True

    if game_state.wait_for_reset and game_state.clock >= 90:
        game_state.wait_for_reset = False
        game_state.clock = -1

    game_state.clock = (game_state.clock + 1) % 100

def check_game_end():
    if game_state.game_mode == "Best of 3" and (game_state.p1_score == 2 or game_state.p2_score == 2):
        return True
    elif game_state.game_mode == "Best of 5" and (game_state.p1_score == 3 or game_state.p2_score == 3):
        return True
    return False

def reset_game():
    global game_state
    current_mode = game_state.game_mode
    game_state = GameState()
    game_state.game_mode = current_mode

def cycle_game_mode():
    modes = ["Normal", "Best of 3", "Best of 5"]
    current_index = modes.index(game_state.game_mode)
    game_state.game_mode = modes[(current_index + 1) % len(modes)]
    reset_game()
    game_state.gameText = f"Game mode changed to: {game_state.game_mode}"

def create_stats_image():
    plt.figure(figsize=(8, 4))  # Adjusted figure size
    plt.clf()
    
    # Player win rates
    total_rounds = len(game_state.round_history)
    p1_wins = sum(1 for _, _, result in game_state.round_history if "Player 1 wins" in result)
    p2_wins = sum(1 for _, _, result in game_state.round_history if "Player 2 wins" in result)
    
    plt.subplot(121)
    plt.pie([p1_wins, p2_wins, total_rounds - p1_wins - p2_wins], 
            labels=['Player 1', 'Player 2', 'Ties'], 
            autopct='%1.1f%%', 
            colors=['#FF9999','#66B2FF','#99FF99'])
    plt.title('Win Distribution')

    # Move frequency
    moves = [move for player in [0, 1] for move, _, _ in game_state.round_history]
    move_counts = Counter(moves)
    
    plt.subplot(122)
    plt.bar(move_counts.keys(), move_counts.values())
    plt.title('Move Frequency')
    plt.ylabel('Count')

    plt.tight_layout()  # Adjust the layout to prevent clipping
    canvas = FigureCanvasAgg(plt.gcf())
    canvas.draw()
    image = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)  # Use buffer_rgba() instead of tostring_rgb()
    image = image.reshape(canvas.get_width_height()[::-1] + (4,))
    image = cv.cvtColor(image, cv.COLOR_RGBA2BGR)
    return image


vid = cv.VideoCapture(0)
width = int(vid.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(vid.get(cv.CAP_PROP_FRAME_HEIGHT))

cv.namedWindow('Advanced Rock Paper Scissors', cv.WINDOW_NORMAL)
cv.resizeWindow('Advanced Rock Paper Scissors', 1280, 720)

with mp_hands.Hands(model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while True:
        ret, frame = vid.read()
        if not ret or frame is None: break

        frame = cv.cvtColor(cv.flip(frame, 1), cv.COLOR_BGR2RGB)
        results = hands.process(frame)
        frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)

        overlay = create_overlay(width, height)
        
        if results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                          mp_drawing_styles.get_default_hand_landmarks_style(),
                                          mp_drawing_styles.get_default_hand_connections_style())
    
                text = f"Player {idx + 1}"
                text_position = (int(hand_landmarks.landmark[0].x * frame.shape[1]), 
                                 int(hand_landmarks.landmark[0].y * frame.shape[0]) - 20)
                if text_position[0] < frame.shape[1] // 2:
                    text_position = (text_position[0] - 80, text_position[1])
                else:
                    text_position = (text_position[0] + 10, text_position[1])
                color = ACCENT_COLOR_1 if idx == 0 else ACCENT_COLOR_2
                draw_modern_text(frame, text, text_position, cv.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        update_game_state(results)

        # Draw modern UI elements
        cv.rectangle(overlay, (0, height - 150), (width, height), BACKGROUND_COLOR, -1)
        cv.addWeighted(overlay, 0.7, frame, 1 - 0.7, 0, frame)

        # Draw game information
        draw_modern_text(frame, f"Time: {int(time.time() - game_state.start_time)}s", (20, height - 130), cv.FONT_HERSHEY_SIMPLEX, 0.7, NEUTRAL_COLOR, 2)
        draw_modern_text(frame, game_state.gameText, (20, height - 90), cv.FONT_HERSHEY_SIMPLEX, 0.9, TEXT_COLOR, 2)
        draw_modern_text(frame, f"Score: Player 1 {game_state.p1_score} - {game_state.p2_score} Player 2", (20, height - 50), cv.FONT_HERSHEY_SIMPLEX, 0.7, TEXT_COLOR, 2)
        draw_modern_text(frame, f"Mode: {game_state.game_mode}", (width - 200, height - 50), cv.FONT_HERSHEY_SIMPLEX, 0.7, ACCENT_COLOR_1, 2)
            
        if check_game_end():
            winner = "Player 1" if game_state.p1_score > game_state.p2_score else "Player 2"
            draw_modern_text(frame, f"{winner} wins the game!", (width // 2 - 150, height // 2), cv.FONT_HERSHEY_SIMPLEX, 1.5, ACCENT_COLOR_1, 3)
            
            # Display stats
            stats_image = create_stats_image()
            stats_height, stats_width = stats_image.shape[:2]
            max_stats_width = width - 100  # Maximum width for stats image
            max_stats_height = height - 100  # Maximum height for stats image
            
            # Resize stats_image if it's too large
            if stats_width > max_stats_width or stats_height > max_stats_height:
                scale = min(max_stats_width / stats_width, max_stats_height / stats_height)
                new_width = int(stats_width * scale)
                new_height = int(stats_height * scale)
                stats_image = cv.resize(stats_image, (new_width, new_height))
                stats_height, stats_width = stats_image.shape[:2]
            
            # Calculate position to center the stats image
            start_x = (width - stats_width) // 2
            start_y = (height - stats_height) // 2
            
            # Create a mask for the stats image region
            mask = np.zeros(stats_image.shape[:2], dtype=np.uint8)
            mask[:, :] = 255
            
            # Create a background with reduced opacity
            overlay = frame.copy()
            cv.rectangle(overlay, (start_x, start_y), (start_x + stats_width, start_y + stats_height), BACKGROUND_COLOR, -1)
            cv.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
            
            # Place the stats image on the frame
            frame[start_y:start_y+stats_height, start_x:start_x+stats_width] = cv.bitwise_and(stats_image, stats_image, mask=mask)

        cv.imshow('Advanced Rock Paper Scissors', frame)


        key = cv.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            reset_game()
        elif key == ord('m'):
            cycle_game_mode()

vid.release()
cv.destroyAllWindows()
import cv2
import numpy as np

img = cv2.imread("tetris.png")
rows, cols, _ = img.shape

# Creating virtual board
virtual_board = np.zeros((rows, cols, 3), dtype=np.uint8)
board_array = np.zeros((20, 10))

# Detecting the board
board_color = np.array([35, 35, 36])
board_mask = cv2.inRange(img, board_color, board_color)
contours, _ = cv2.findContours(board_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)

cnt = contours[0]
(board_x, board_y, board_w, board_h) = cv2.boundingRect(cnt)
cv2.drawContours(img, [cnt], -1, (0, 255, 0), 3)
cv2.drawContours(virtual_board, [cnt], -1, (0, 255, 0), 3)





# Detecting Tetrominoes
tetrominoes = {"i_polyomino": [116, 98, 0],
               "o_polyomino": [0, 102, 116],
               "t_polyomino": [127, 0, 106],
               "j_polyomino": [127, 67, 0],
               "l_polyomino": [0, 85, 127],
               "s_polyomino": [35, 127, 0],
               "z_polyomino": [0, 0, 116]}

# Creating a mask for each tetromino
for key in tetrominoes:
    bgr_color = tetrominoes[key]
    bgr_color = np.array(bgr_color)
    mask = cv2.inRange(img, bgr_color, bgr_color)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        (x, y, w, h) = cv2.boundingRect(cnt)
        cv2.rectangle(virtual_board, (x, y), (x + w, y + h),
                      (0, 0, 255), 2)

        # Creating cells into the board
        block_width = int(board_w / 10)
        block_height = int(board_h / 20)

        for board_row in range(20):
            for board_col in range(10):
                block_x = block_width * board_col
                block_y = block_height * board_row

                cv2.rectangle(virtual_board, (board_x + block_x, board_y + block_y),
                              (board_x + block_x + block_width, board_y + block_y + block_height),
                              (255, 255, 255), 1)


                # Check if tetrominoe is inside the cell
                if board_x + block_x <= x < board_x + block_x + block_width and board_y + block_y <= y < board_y + block_y + block_height:

                    board_array[board_row, board_col] = 1




        cv2.imshow("virtula board", virtual_board)
        #cv2.waitKey(0)




i_polyomino = np.array([116, 98, 0])
i_polyomino_mask = cv2.inRange(img, i_polyomino, i_polyomino)

print(board_array)
cv2.imshow("Mask", mask)
cv2.imshow("I polyomino mask", i_polyomino_mask)
cv2.imshow("Img tetris", img)
cv2.imshow("Virtual board", virtual_board)
cv2.waitKey(0)
cv2.destroyAllWindows()
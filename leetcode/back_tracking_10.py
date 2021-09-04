from typing import List


class Solution:
    # 37. Sudoku Solver  back tracking
    # t: O(9!)*9   s:O(81)
    def solveSudoku(self, board: List[List[str]]):
        rows, cols = [set() for _ in range(len(board))], [set() for _ in range(len(board))]
        boxes = [set() for _ in range(10)]
        for i in range(len(board)):
            for j in range(len(board[0])):
                if board[i][j] != ".":
                    rows[i].add(board[i][j])
                    cols[j].add(board[i][j])
                    boxes[self.find_box(i, j)].add(board[i][j])

        self.solve(0, 0, board, rows, cols, boxes)

    @staticmethod
    def find_box(i, j):
        row, col = i // 3, j // 3
        return row * 3 + col

    def unfill(self, row, col, board, rows, cols, boxes):
        if board[row][col] != ".":
            val = board[row][col]
            board[row][col] = "."
            rows[row].remove(val)
            cols[col].remove(val)
            boxes[self.find_box(row, col)].remove(val)

    def fill(self, row, col, board, rows, cols, boxes, val):
        if board[row][col] == ".":
            if val not in rows[row] and val not in cols[col] and val not in boxes[self.find_box(row, col)]:
                rows[row].add(val)
                cols[col].add(val)
                boxes[self.find_box(row, col)].add(val)
                board[row][col] = val
                return True
        return False

    def solve(self, row, col, board, rows, cols, boxes):
        if col >= len(board[0]):
            col = 0
            row += 1
        if row >= len(board):
            return True
        if board[row][col] == ".":
            for i in range(1, len(board) + 1):
                res = self.fill(row, col, board, rows, cols, boxes, str(i))
                if res:
                    if self.solve(row, col + 1, board, rows, cols, boxes):
                        return True
                    self.unfill(row, col, board, rows, cols, boxes)
        else:
            if self.solve(row, col + 1, board, rows, cols, boxes):
                return True
        return False

    # 17. Letter Combinations of a Phone Number   backtracking/dfs  t: O(4^len)
    def letterCombinations(self, digits: str) -> List[str]:
        self.dic = {"2": ["a", "b", "c"], "3": ["d", "e", "f"], "4": ["g", "h", "i"], "5": ["j", "k", "l"],
                    "6": ["m", "n", "o"], "7": ["p", "q", "r", "s"], "8": ["t", "u", "v"], "9": ["w", "x", "y", "z"], }
        res = []
        if not len(digits):
            return res
        self.helper(0, digits, "", res)

        return res

    def helper(self, index, digits, temp, res):
        if index >= len(digits):
            res.append(temp)
            return

        for j in self.dic.get(digits[index]):
            """
            temp = temp + j
            self.helper(index+1, digits, temp, res)
            temp = temp[:-1]
            """
            self.helper(index + 1, digits, temp + j, res)


if __name__ == '__main__':
    solution = Solution()
    solution.solveSudoku([["5", "3", ".", ".", "7", ".", ".", ".", "."], ["6", ".", ".", "1", "9", "5", ".", ".", "."],
                          [".", "9", "8", ".", ".", ".", ".", "6", "."], ["8", ".", ".", ".", "6", ".", ".", ".", "3"],
                          ["4", ".", ".", "8", ".", "3", ".", ".", "1"], ["7", ".", ".", ".", "2", ".", ".", ".", "6"],
                          [".", "6", ".", ".", ".", ".", "2", "8", "."], [".", ".", ".", "4", "1", "9", ".", ".", "5"],
                          [".", ".", ".", ".", "8", ".", ".", "7", "9"]])

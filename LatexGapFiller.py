from abc import ABC, abstractmethod
import array_to_latex as a2l
import numpy as np

class Sheet(ABC):
    
    def __init__(self, body) -> None:
        # self._id = id
        self._body = body
        self.generated = None

    # @property
    # def get_id(self) -> int:
    #     return self._id
    
    @property
    def get_body(self):
        return self._body
    
    @property
    def set_body(self, body) -> None:
        self._body = body


class TextSheet(Sheet):

    def __init__(self, body:str = "") -> None:
        with open(body, 'r') as file:
            body = file.read()
        self.key_words = dict()
        super().__init__(body)

        self.__parse_variables()

    def __parse_variables(self):
        balance = 0
        key_word = ""
        for i in range(len(self._body)):
            
            if self._body[i] == '}' and balance == 1:
                balance -= 1
                self.key_words[str(key_word)] = None
                key_word = ""
            
            if balance == 1:
                key_word += self._body[i]

            if self._body[i] == '{' and (i == 0 or self._body[i - 1] == ' '):
                balance += 1

            elif balance < 0 or balance > 1:
                raise RuntimeError(f"Unpair bracket with index {i}, cannot parse text!")


    def fill_gap(self, key_words):
        filled_body = ''.join(self._body)
        for key in key_words.keys():
            filled_body = filled_body.replace('{' + key + '}', str(key_words[key]))
            # print('{' + key + '}', str(key_words[key]))
        self.generated = filled_body


class LatexGapFiller():

    def __init__(self) -> None:
        self.__key_words = dict()
        self.__sheets = []

    def add_sheet(self, sheet: TextSheet):
        sheet.fill_gap(self.__key_words)
        self.__sheets.append(sheet.generated)


    def add_key_word(self, name :str, value):
        # variable_name = f'{site=}'.split('=')[0]
        value = np.array(value)
        if value.flatten().shape[0] > 1:
            value = np.array(value)
            value = np.where(value.imag == 0, value.real, value)
                # if value[i].imag == 0:
                #     value[i] = value[i].real
            ans =  str(a2l.to_ltx(value, frmt = '{:2.2f}', arraytype = 'bmatrix', print_out=False))
            ans = ans.replace('.00', '')
            ans = ans.replace('.j', 'j')
            ans = ans.replace('+  0j', '')
            ans = ans.replace('+ 0j', '')
            ans = ans.replace('+0j', '')
            ans = ans.replace('+-', '-')
            ans = ans.replace('+ -', '-')
            self.__key_words[name] = ans
        else:
            self.__key_words[name] =  str(value.flatten()[0])

    def generate_text(self, file_name = 'report.txt'):
        text_file = open(file_name, "w")
        for sheet in self.__sheets:
            text_file.write(sheet + 2*'\n')
            

if __name__ == '__main__':
    with open('./lab8/task1.txt', 'r') as file:
        data = file.read()

    gap_filler = LatexGapFiller()

    for i in range(2):
        text = TextSheet('./lab8/task1.txt')
        gap_filler.add_key_word("A", i)
        gap_filler.add_sheet(text)

    gap_filler.generate_text()
       

        
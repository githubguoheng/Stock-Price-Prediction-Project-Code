import os

with open("zhaoshang", "r") as file:
    lines = file.readlines()
    i = 0
    newLines = []
    newLine = ''
    
    for line in lines:
        line = line.strip("\n")

        if  line.find("/") != -1:
            line = line.replace("/","-")
        if line.find("+") != -1:
            line = line.replace("+","")

        if line.find("万") != -1:
            line = line.replace("万","")
            line = str(round(float(line) * 10000, 1))
        elif line.find("亿") != -1:
            line = line.replace("亿","")
            line = str(round(float(line) * 100000000, 1))
        elif line.find("%") != -1:
            line = line.replace("%","")
            line = str(round(float(line) / 100, 4))            

        if i % 8 == 7:
            newLine = newLine + line + "\n"
            newLines.append(newLine)
            newLine = ''
        elif i % 8 < 7:
            line = line + ","
            newLine = newLine + line
        i = i + 1
        # print(line)
    # print(newLines)

    # print(newLines[-1:-len(newLines):-1])

    sortedLines = newLines[-1:-len(newLines):-1]

    # print(newLines[0])

    sortedLines.insert(0, newLines[0])

    print(sortedLines)


with open("China Merchants Bank.csv", "w") as file:
    file.writelines(sortedLines)

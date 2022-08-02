import csv

with open("conv2d_top10_flops.txt", "r") as readFile:
    rows = readFile.readlines()
    # print(len(rows))
    # for i in range(len(rows)):
    #     print(float(rows[i]))
    with open("conv2d_top10_flops.csv", "w") as writeCSV:
        writer = csv.writer(writeCSV)
        with open("conv2d_top10.csv", "r") as readCSV:
            info_rows = csv.reader(readCSV)
            i = 0
            for row in info_rows:
                if i == 0:
                    row.append("Mflops")
                else:
                    row.append(float(rows[i-1]))
                writer.writerow(row)
                i += 1
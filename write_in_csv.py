from csv import writer

def write_attendence(name, curr_datetime):
    List=[name,curr_datetime]
    with open('attendence.csv', 'a') as f_object:
        writer_object = writer(f_object)
        writer_object.writerow(List)
        f_object.close()




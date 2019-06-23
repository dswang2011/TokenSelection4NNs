

def write_line(content,file_path):
	with open(file_path,'a',encoding='utf8') as f:
		f.write(content.strip())
		f.write('\n')
		f.close()



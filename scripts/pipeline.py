import json
import pymongo
import random
from tqdm import tqdm

class Reader:
        
    def __iter__(self):
        '''
        All reader must implement the "__iter__" method, it read a batch each time.
        '''
        raise NotImplementedError

class FileReader(Reader):
    
    def __init__(self, filename, batch_size=1, limit=-1):
        self.filename = filename
        self.batch_size = batch_size
        self.limit = limit
    
    def __iter__(self):
        with open(self.filename) as fp:
            lines = []
            count = 0
            for line in fp:
                lines.append(line)
                if len(lines) >= self.batch_size:                    
                    yield lines
                    count += len(lines)
                    if self.limit > 0 and count > self.limit:
                        break
                    lines = []
            if len(lines) > 0:
                yield lines


class MongoReader(Reader):
    
    def __init__(self, host, port, db, collection, username=None, password=None, batch_size=1, limit=-1, query={}):
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.db = db
        self.collection = collection
        self.batch_size = batch_size
        self.limit = limit
        self.visited = 0
        self.query = query
        
    def __iter__(self):
        client = pymongo.MongoClient(host=self.host, port=self.port, username=self.username, password=self.password)
        collection = client[self.db][self.collection]
        total = collection.find(self.query).count()
        if self.limit > 0 and total > self.limit:
            total = self.limit

        records = []
        while self.visited < total:
            for r in collection.find().skip(self.visited).limit(self.batch_size):
                records.append(r)
            yield records
            self.visited += len(records)
            records = []
        client.close()
    

class Writer:
    
    def __call__(self, data):
        '''
        Write a batch at a time
        '''
        print(data)

class JsonlWriter(Writer):
    
    def __init__(self, filename):
        self.filename = filename
        
    def __call__(self, data):
        with open(self.filename, 'w') as fp:            
            for item in data:
                fp.write(json.dumps(item) + '\n')
                
class TxtWriter(Writer):
    
    def __init__(self, filename):
        self.filename = filename

    def prepro(self, data):
        return data

    def __call__(self, data):
        with open(self.filename, 'w') as fp:
            self.write(fp, data)

    def write(self, fp, data):
        for item in data:
            fp.write(item.strip() + '\n')

class ConllWriter(TxtWriter):
    '''
    write annotations in Conll2003 format: word, pos, chunk, ner
    '''
    def prepro(self, data):
        random.shuffle(data)
        return data
        
    def write(self, fp, data):
        fp.write('-DOCSTART-\tpos\tchunk\tner\n\n')
        for aligned in data:
            for word, tag in aligned:
                fp.write('%s\tx\tx\t%s\n' % (word, tag))
            fp.write('\n')
                    
class MongoWriter(Writer):
    
    def __init__(self, host, port, db, collection, username=None, password=None):
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.db = db
        self.collection = collection        
        
    def __call__(self, batch):
        client = pymongo.MongoClient(host=self.host, port=self.port, username=self.username, password=self.password)
        collection = client[self.db][self.collection]
        result = []
        for item in batch:
            result.append(self.update(collection, item))
        client.close()
        return []
    
    def update(self, collection, item):
        print('update: ', item['document'], item['score'])
        return collection.replace_one({'_id': item['_id']}, item)


class Processor:
    
    def __init__(self, ops = None):
        self.ops = ops or []
        
    def add(self, op):
        self.ops.append(op)

    def __call__(self, batch):
        return batch


class InstanceProcessor(Processor):
    '''
    A map function, apply all processors on each instance
    '''
    def __call__(self, batch):
        '''
        apply operators on each instance
        '''        
        old_data, new_data = batch, []
        for op in self.ops:
            for item in old_data:
                new_item = op(item)
                if new_item is not None:
                    new_data.append(new_item)
            old_data, new_data = new_data, []
        return old_data

class BatchProcessor(Processor):
    '''
    A reduce function, apply the operatos on the whole batch
    '''
    def __call__(self, batch):
        old_batch, new_batch = batch, None
        for op in self.ops:
            new_batch = op(old_batch)
            old_batch, new_batch = new_batch, None
        return old_batch
    
class Pipeline:
    
    def __init__(self, reader, mappers = [], reducers = [], writer = None):
        self.reader = reader
        self.writer = writer
                
        self.mapper = InstanceProcessor(mappers)
        self.reducer = BatchProcessor(reducers)
    
    def transform(self, batch):
        result = self.mapper(batch)
        return self.reducer(result)
    
    def run(self):
        result = []
        for batch in self.reader:
            result.extend(self.transform(batch))
            
        if self.writer:
            return self.writer(result)
        return result

class LineOps:
    '''
    Assume the input is a string line, the output can be a string, a list or a dict object
    '''
    @staticmethod
    def check_data(line):
        assert(isinstance(line, str))
        
    @staticmethod
    def parse_json(line):
        LineOps.check_data(line)
        return json.loads(line)

    @staticmethod
    def split(line):
        LineOps.check_data(line)
        items = line.strip().split()
        return items

    @staticmethod
    def limit_maxlen(line):
        LineOps.check_data(line)
        return line[:100]
    
    @staticmethod
    def remove_tail_marks(line):
        LineOps.check_data(line)
        return line.strip()
    
    @staticmethod
    def remove_punctations(line):
        LineOps.check_data(line)
        return line.replace('!.,', '')

class ListOps:
    
    @staticmethod
    def check_data(line):
        assert(isinstance(line, list))
    
    @staticmethod
    def parse_json(arr):
        ListOps.check_data(arr)
        return [json.loads(line) for line in arr]

    @staticmethod
    def build_triples(arr):
        ListOps.check_data(arr)
        if len(arr) == 3:
            return {
                'h': arr[0],
                't': arr[1],
                'l': arr[2]
            }
        return None
    
    @staticmethod
    def flatten(arr):
        ListOps.check_data(arr)
        return [w for item in arr for w in item]
    
    @staticmethod
    def inspect(arr):
        ListOps.check_data(arr)
        print(len(arr), type(arr[0]))
        return arr
    
    @staticmethod
    def shuffle(arr):
        ListOps.check_data(arr)
        random.shuffle(arr)
        return arr

class DictOps:
    
    @staticmethod
    def maxlen_sent(data):
        if data.get('sent'):
            data['sent'] = data['sent'][:50]
        return data


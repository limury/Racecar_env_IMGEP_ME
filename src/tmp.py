import numpy as np

class MetricUtils:
    @staticmethod
    def write_array(a, f):
        for i in a:
            f.write(str(i) + ',')

    @staticmethod
    def save_archive(archive, gen, log_dir, presets, params):
        generation = str(gen)
        generation = '0'*(8-len(generation)) + generation
        filename = log_dir+'/' + presets + generation + '.dat'

        data = list(archive.values())
        metrics = []
        if ('genotype' in params['log_outputs']):
            genos = np.vectorize(lambda x: x.x)
            metrics.append( genos(data) )

        if ('BD' in params['log_outputs']):
            desc = np.vectorize(lambda x: x.desc)
            metrics.append( desc(data) )
            
        if ('fitness' in params['log_outputs']):
            fitness = np.vectorize(lambda x: x.fitness)
            metrics.append( fitness(data) )

        if ('centroid' in params['log_outputs']):
            centroid = np.vectorize(lambda x: x.centroid)
            metrics.append( np.stack(centroid(data), axis=1) )
        
        if ('curiosity' in params['log_outputs']):
            curiosity = np.vectorize(lambda x: x.curiosity)
            metrics.append( np.expand_dims(curiosity(data), axis=1) )
        
        metrics = np.concatenate(metrics, axis=1)

        #out = []
        #for x in archive.values():
            #out.append( [x.centroid[0], x.centroid[1], x.centroid[2], x.curiosity] )

        #val = True
        #for x, y in zip(metrics, out):
            #for a, b in zip(x, y):
                #if (a!=b):
                    #val = False
        #print("Are the arrays the same? ", val)


        with open(filename, 'w') as f:
            np.savetxt(filename, metrics, delimiter=',', fmt='%f')

















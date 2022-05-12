# ==============================================================================
# ==============================================================================
# 
# This script contains functions for cross matching mergen classifications with
# stellar variabiilty catalogs (e.g. ASAS-SN, Simbad, GCVS)
# 
# // Emma Chickles
# 
# -- existing functions --------------------------------------------------------
# 
# querying catalogs:
# * query_simbad 
# * query_gcvs
# * query_asas_sn
# 
# organizing object types:
# * get_var_descr
# * make_parent_dict
# * make_variability_tree
# * make_redundant_otype_dict
# * merge_otype
# * get_parent_otypes
# * get_parents_only
# * make_remove_class_list
# * make_flagged_class_list
# * write_true_label_txt
# * make_true_label_txt
# * read_otype_txt
# * quick_simbad
# * get_true_var_types
# 
# -- TODO ----------------------------------------------------------------------
# 
# * clean up organization fcns
#   * get_var_hier fcn
# * deal with dependencies on data_dir (clarify what is required to run these
#   functions)
#   * should there be a function that sets up the folder structure? or should
#   * that happen in mergen?
# * currently only handles the 2 minute cadence TICIDs (fast cadence during 
#   Cycles 1 and 2)
# 
# ==============================================================================
# ==============================================================================

from __init__ import *

import astropy.coordinates as coord
import astropy.units as u
from astroquery.simbad import Simbad

# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# :: Query catalogs ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

def query_catalogs(metapath, savepath,  sector='all', query_mast=False,
                   catalogs=['simbad', 'asassn']):
    if 'simbad' in catalogs:
        query_simbad(metapath, savepath, sector, query_mast)
        correct_simbad_to_vizier(metapath)
    if 'gcvs' in catalogs:
        query_gcvs(metapath, sector)
    if 'asassn' in catalogs:
        query_asas_sn(metapath, savepath, sector)

def query_simbad(metapath, savepath, sector='all', query_mast=False, sep=','):
    '''Cross-matches ASAS-SN catalog with TIC catalog based on matching GAIA IDs
    * data_dir
    * sector: 'all' or int, currently only handles short-cadence'''

    # import time
    
    customSimbad = Simbad()
    customSimbad.add_votable_fields('otypes')
    # customSimbad.add_votable_fields('biblio')

    if sector=='all':
        sectors = list(range(1,27))
    else:
        sectors=[sector]

    for sector in sectors:
        print('Sector '+str(sector))
        out_f = metapath+'spoc/cat/sector-%02d'%sector+'_simbad_raw.txt'

        with open(out_f, 'w') as f:
            f.write('TICID'+sep+'TYPE'+sep+'MAIN_ID\n')
            # f.write(align%('TICID', 'TYPE', 'MAIN_ID')+'\n')

        ticid_already_classified = []
        # if not os.path.exists(out_f):
        #     with open(out_f, 'w') as f:
        #         f.write('TICID\tTYPE\tMAIN_ID\n')
        # with open(out_f, 'r') as f:
        #     lines = f.readlines()
        #     ticid_already_classified = []
        #     for line in lines[1:]:
        #         ticid_already_classified.append(float(line.split(',')[0]))

        if not query_mast:
            tic_cat=pd.read_csv(metapath+'spoc/tic/sector-%02d'%sector+\
                                     '-tic_cat.csv')
            ticid_list = tic_cat['ID']
            # sector_data = np.loadtxt(metapath+'spoc/targ/2m/'+\
            #                          'all_targets_S%03d'%sector+'_v1.txt')
            # ticid_list = sector_data[:,0]

        print(str(len(ticid_list))+' targets')
        print(str(len(ticid_already_classified))+' targets completed')
        ticid_list = np.setdiff1d(ticid_list, ticid_already_classified)
        print(str(len(ticid_list))+' targets to query')

        count = 0
        for tic in ticid_list:

            count += 1
            res = None

            n_iter = 0
            while res is None:
                try:
                    print(str(count)+'/'+str(len(ticid_list))+\
                          ': finding object type for Sector ' +str(sector)+\
                          ' TIC' + str(int(tic)))

                    target = 'TIC ' + str(int(tic))                    
                    if query_mast:
                        # >> get coordinates
                        catalog_data = Catalogs.query_object(target, radius=0.02,
                                                             catalog='TIC')[0]

                    else:
                        ind = np.nonzero(tic_cat['ID'].to_numpy() == tic)[0][0]
                        catalog_data=tic_cat.iloc[ind]
                    # time.sleep(6)


                    # -- get object type from Simbad ---------------------------

                    # >> first just try querying Simbad with the TICID
                    res = customSimbad.query_object(target)
                    # time.sleep(6)

                    # >> if no luck with that, try checking other catalogs
                    catalog_names = ['TYC', 'HIP', 'TWOMASS', 'SDSS', 'ALLWISE',
                                     'GAIA', 'APASS', 'KIC']
                    for name in catalog_names:
                        if type(res) == type(None):
                            if type(catalog_data[name]) != np.ma.core.MaskedConstant:
                                target_new = name + ' ' + str(catalog_data[name])
                                res = customSimbad.query_object(target_new)
                                # time.sleep(6)

                    # time.sleep(6)
                except:
                    pass
                    print('connection failed! Trying again now')

                n_iter += 1

                if n_iter > 5: 
                    break

            # -- save results --------------------------------------------------
            if type(res) == type(None):
                print('failed :(')
                res=0 
                with open(out_f, 'a') as f:
                    line = '{}'+sep+'NONE'+sep+'NONE\n'
                    f.write(line.format(tic))
                    # f.write(align%(tic, '', '')+'\n')

            else:
                # otypes = res['OTYPES'][0].decode('utf-8')
                otypes = res['OTYPES'][0]
                # main_id = res['MAIN_ID'].data[0].decode('utf-8')
                main_id = res['MAIN_ID'][0]

                with open(out_f, 'a') as f:
                    line = '{}'+sep+'{}'+sep+'{}\n'
                    f.write(line.format(tic, otypes, main_id))
                    # f.write(align%(tic, otypes, main_id)+'\n')

            
def query_gcvs(metapath, savepath, sector='all', tol=0.1, diag_plot=True):
    '''Cross-matches GCVS catalog with TIC catalog.
    * data_dir
    * sector: 'all' or int, currently only handles short-cadence
    * tol: maximum separation of TIC target and GCVS target (in arcsec)

    See column names and descriptions of gcvs5.txt here:
    https://heasarc.gsfc.nasa.gov/W3Browse/star-catalog/gcvs.html
    '''
    # data = pd.read_csv(metapath+'gcvs5.txt', delimiter='|')

    gcvs_id = []
    vartype = []
    min_mag = []
    max_mag = []
    ra_all, dec_all = [], []
    with open(metapath+'gcvs5.txt', 'r') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            if i % 500 == 0: print(str(i)+' / '+str(len(lines)))
            line = lines[i]
            radec = line.split('|')[2].split(' ')
            while '' in radec:
                radec.remove('')
            if len(radec) != 0:
                gcvs_id.append(line.split('|')[1])
                vartype.append(line.split('|')[3].split(' ')[0])

                # >> coordinates
                ra = radec[0][:2]+' '+radec[0][2:4]+' '+radec[0][4:]
                dec = radec[1][:3]+' '+radec[1][3:5]+' '+radec[1][5:]
                ra_all.append(ra)
                dec_all.append(dec)

                # >> minimum magnitude
                flags = ['<', '>', '(', ')', ':', 'R', 'c', 'b', 'B', 'V', 'I',
                         'u', 'U', "'", 'v', 'y', 'i', 'p', 'g', '*', ' ']
                obj_mag = line.split('|')[4]
                for flag in flags:
                    while flag in obj_mag:
                        ind = obj_mag.index(flag)
                        obj_mag = obj_mag[:ind]+obj_mag[1+ind:]
                if len(obj_mag) == 0:
                    obj_mag = np.nan
                min_mag.append(float(obj_mag))
                
                # >> maximum magnitude 
                obj_mag = line.split('|')[5]
                for flag in flags:
                    while flag in obj_mag:
                        ind = obj_mag.index(flag)
                        obj_mag = obj_mag[:ind]+obj_mag[1+ind:]
                if len(obj_mag) == 0:
                    obj_mag = np.nan
                max_mag.append(float(obj_mag))

    data_coords = coord.SkyCoord(ra=ra_all, dec=dec_all,
                                 unit=(u.hourangle, u.deg))

    if sector=='all':
        sectors = list(range(1,27))
    else:
        sectors=[sector]

    for sector in sectors:
        out_fname = metapath+'spoc/cat/sector-%02d'%sector+'_gcvs_raw.txt'
        sector_data = np.loadtxt(metapath+'spoc/targ/2m/'+\
                                 'all_targets_S%03d'%sector+'_v1.txt')
        print('Loaded Sector'+str(sector)+'tic_cat_all.csv')

        # >> find GCVS target closest to each TIC target
        min_sep = []
        min_inds = []
        for i in range(len(sector_data)):
            if i % 500 == 0:
                print('TIC '+str(int(sector_data[i,0]))+'\t'+str(i)+'/'+\
                      str(len(sector_data)))
            ticid_coord = coord.SkyCoord(sector_data[i, 4],
                                         sector_data[i, 5],
                                         unit=(u.deg, u.deg)) 
            idx, d2d, d3d =  ticid_coord.match_to_catalog_3d(data_coords)
            min_sep.append(d2d)
            min_inds.append(idx)

        sep_arcsec = np.array([sep.to(u.arcsec).value[0] for sep in min_sep])
        min_inds = np.array(min_inds).astype('int')

        # >> save the variability type if GCVS target is close enough
        with open(out_fname, 'w') as f:

            for i in range(len(sector_data)):
                if sep_arcsec[i] < tol:
                    ind = min_inds[i]            
                    f.write(str(int(sector_data[i,0]))+','+\
                            str(vartype[ind])+','+\
                            str(gcvs_id[ind])+'\n')

                else:
                    f.write(str(int(sector_data[i,0]))+',NONE,NONE\n')

        # >> plotting
        if diag_plot:
            # >> make histogram of minimum separations
            fig, ax = plt.subplots()
            bins = 10**np.linspace(np.floor(np.log10(np.nanmin(sep_arcsec))),
                                   np.ceil(np.log10(np.nanmax(sep_arcsec))), 50)
            ax.hist(sep_arcsec, bins=bins, log=True)
            ax.set_xlabel('separation (arcseconds)')
            ax.set_ylabel('number of targets in Sector '+str(sector))
            ax.set_xscale('log')
            fig.savefig(savepath+'sector%02d'%sector+'_gcvs_sep_arcsec.png')
            print('Saved '+savepath+'sector%02d'%sector+'_gcvs_sep_arcsec.png')
            plt.close()
            
            # >> compare magnitude from TIC and ASAS-SN of cross-matched targets
            tol_tests = [10, 1, 0.1] 
            for tol in tol_tests:
                inds1 = np.nonzero(sep_arcsec < tol)
                inds2 = min_inds[inds1]
                print('Tolerance: '+str(tol)+' arcseconds, number of targets: '+\
                      str(len(inds1[0])))
                plt.figure()
                plt.plot(sector_data[inds1[0], 3],
                         np.array(min_mag)[inds2], '.k')
                plt.xlabel('TESS magnitude')
                plt.ylabel('Minimum GCVS magnitude')
                out_f = savepath+'sector%02d'%sector+'_gcvs_tol'+\
                        str(tol)+'_min.png'
                plt.savefig(out_f)
                print('Saved '+out_f)
                plt.close()

                plt.figure()
                plt.plot(sector_data[inds1[0], 3],
                         np.array(max_mag)[inds2], '.k')
                plt.xlabel('TESS magnitude')
                plt.ylabel('Maximum GCVS magnitude')
                out_f = savepath+'sector%02d'%sector+'_gcvs_tol'+\
                        str(tol)+'_max.png'
                plt.savefig(out_f)
                print('Saved '+out_f)
                plt.close()
                
def query_asas_sn(metapath, savepath, sector='all', diag_plot=True,
                  use_sep=False):
    '''Cross-matches ASAS-SN catalog with TIC catalog based on matching GAIA IDs
    * data_dir
    * sector: 'all' or int, currently only handles short-cadence
    
    Can read output txt file with
    np.loadtxt('sector-01_asassn_raw.txt', delimiter=',', dtype='str',
               skiprows=1)
    '''
    # data = pd.read_csv(data_dir+'asas_sn_database.csv')
    data = pd.read_csv(metapath+'asassn_catalog.csv')
    print('Loaded asas_sn_database.csv')
    data_coords = coord.SkyCoord(data['raj2000'], data['dej2000'],
                                 unit=(u.deg, u.deg))
    ticid = data['tic_id']
    nan_inds = ['TIC ' in str(i) for i in ticid]
    ticid = np.array(ticid)[np.nonzero(np.array(nan_inds))]
    ticid = np.array([int(float(i[4:])) for i in ticid])

    if sector=='all':
        sectors = list(range(1,27))
    else:
        sectors=[sector]

    for sector in sectors:
        # sector_data = pd.read_csv(metapath+'spoc/tic/sector-%02d'%sector+\
        #                           '-tic_cat.csv')
        sector_data = np.loadtxt(metapath+'spoc/targ/2m/'+\
                                 'all_targets_S%03d'%sector+'_v1.txt')
        out_fname = metapath+'spoc/cat/sector-%02d'%sector+'_asassn_raw.txt'

        _, comm1, comm2 = np.intersect1d(sector_data[:,0], ticid,
                                         return_indices=True)

        # >> save cross-matched target in text file
        with open(out_fname, 'w') as f:
            f.write('TICID,TYPE,ASASSN_NAME\n')
            # f.write(align%('TICID', 'TYPE', 'ASASSN_NAME')+'\n')
            for i in range(len(sector_data)):            
                if i in comm1:
                    ind = comm2[np.nonzero(comm1 == i)][0]
                    f.write(str(int(sector_data[i,0]))+','+\
                            str(data['variable_type'][ind])+','+\
                            str(data['asassn_name'][ind])+'\n')

                    # f.write(align%(int(sector_data[i,0]), \
                    #                data['variable_type'][ind],\
                    #                data['asassn_name'][ind])+'\n')
                else:
                    f.write(str(int(sector_data[i,0]))+',NONE,NONE\n')
                    # f.write(align%(int(sector_data[i,0]),'NONE','NONE')+'\n')
        print('Saved '+out_fname)

        if diag_plot:
            # >> compare magnitude from TIC and ASAS-SN of cross-matched targets
            plt.figure()
            plt.plot(sector_data[:,3][comm1], data['mean_vmag'][comm2], '.k')
            plt.xlabel('TESS magnitude (TIC)')
            plt.ylabel('Mean Vmag (ASAS-SN)')
            plt.savefig(savepath+'cat/sector-%02d'%sector+\
                        '_asassn_mag_cross_match.png')
            plt.close()

            plt.figure()
            plt.plot(sector_data[:,4][comm1], data['raj2000'][comm2], '.k')
            plt.xlabel('RA (TIC)')
            plt.ylabel('RA (ASAS-SN)')
            plt.savefig(savepath+'cat/sector-%02d'%sector+\
                        '_asassn_ra_cross_match.png')
            plt.close()

            plt.figure()
            plt.plot(sector_data[:,5][comm1], data['dej2000'][comm2], '.k')
            plt.xlabel('DEC (TIC)')
            plt.ylabel('DEC (ASAS-SN)')
            plt.savefig(savepath+'cat/sector-%02d'%sector+\
                        '_asassn_de_cross_match.png')
            plt.close()

            if use_sep:
                # >> get minimum separations between TIC and ASAS-SN targts
                if os.path.exists(savepath+'cat/asassn_sep.txt'):
                    sep_arcsec = np.loadtxt(savepath+'cat/asassn_sep.txt')
                    min_inds = np.loadtxt(savepath+\
                                          'cat/asassn_sep_inds.txt').astype('int')
                else:
                    min_sep = []
                    min_inds = []
                    for i in range(len(sector_data)):
                        print('TIC '+str(int(sector_data[i,0]))+'\t'+str(i)+\
                              '/'+str(len(sector_data)))
                        ticid_coord = coord.SkyCoord(sector_data[i,4],
                                                     sector_data[i,5],
                                                     unit=(u.deg, u.deg)) 
                        sep = ticid_coord.separation(data_coords)
                        min_sep.append(np.min(sep))
                        min_inds.append(np.argmin(sep))
                    sep_arcsec = np.array([sep.to(u.arcsec).value for sep in min_sep])
                    min_inds = np.array(min_inds)
                    np.savetxt(savepath+'cat/asassn_sep.txt', sep_arcsec)
                    np.savetxt(savepath+'cat/asassn_sep_inds.txt', min_inds)

                # >> make histogram of minimum separations
                fig, ax = plt.subplots()
                ax.hist(sep_arcsec, bins=10**np.linspace(-2, 4, 30), log=True)
                ax.set_xlabel('arcseconds')
                ax.set_ylabel('number of targets in Sector '+str(sector))
                ax.set_xscale('log')
                fig.savefig(savepath+'cat/asassn_sep_arcsec.png')

                fig1, ax1 = plt.subplots()
                ax1.hist(sep_arcsec[comm1], bins=10**np.linspace(-2, 4, 30), log=True)
                ax1.set_xlabel('arcseconds')
                ax1.set_ylabel('number of cross-matched targets in Sector '+str(sector))
                ax1.set_xscale('log')
                ax1.set_ylim(ax.get_ylim())
                fig1.savefig(savepath+'cat/asassn_sep_cross_match.png')

                # >> compare magnitude from TIC and ASAS-SN of cross-matched targets
                tol_tests = [10, 1, 0.1]
                for tol in tol_tests:
                    inds1 = np.nonzero(sep_arcsec < tol)
                    inds2 = min_inds[inds1]
                    plt.figure()
                    plt.plot(sector_data[:,3][inds1][0],
                             data['mean_vmag'][inds2], '.k')
                    plt.xlabel('TESS magnitude (TIC)')
                    plt.ylabel('Mean Vmag (ASAS-SN)')
                    plt.savefig(savepath+'cat/asassn_mag_tol'+str(tol)+'.png')
                    plt.close()

# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# :: Clean object types ::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

def clean_simbad(metapath,
                 var_simbad='tess_stellar_var/docs/var_simbad.txt',
                 uncertainty_flags=[':', '?', '*']):
    '''Removes types unrelated to variability, and uses Vizier acronyms instead
    of SIMBAD acronyms.'''
    # -- create dictionary from Simbad to GCVS labels --------------------------
    with open(var_simbad, 'r') as f:
        lines = f.readlines()
    renamed = {}
    for line in lines[1:]: 
        otype, description = line.split(',')
        description = description.replace('\n', '') # >> rmv new line char        
        renamed[otype] = description 

    # -- classes to be removed -------------------------------------------------
    remove_classes = make_remove_class_list(catalog='simbad')

    # -- produce revised cross-matched classifications -------------------------

    fnames = [f for f in os.listdir(metapath+'spoc/cat/') if 'simbad_raw' in f]
    fnames.sort()

    for fname in fnames:
        # filo = pd.read_csv(metapath+'spoc/cat/'+fname, delimiter='\s+,')
        filo = np.loadtxt(metapath+'spoc/cat/'+fname, delimiter=',',
                          dtype='str', skiprows=1)
        # >> original fname ends with '_simbad_raw.txt'
        # >> output fname ends with '_simbad_cln.txt'
        out_f = metapath+'spoc/cat/'+fname[:-8]+'_cln.txt' 
        with open(out_f, 'w') as f:
            f.write('TICID,TYPE,MAIN_ID'+'\n')
            # f.write(align%('TICID','TYPE','MAIN_ID')+'\n')

        for i in range(len(filo)):
            tic = filo[i][0]
            otype = filo[i][1]

            main = filo[i][2]
            # tic = filo.iloc[i]['TICID']
            # otype = filo.iloc[i]['TYPE']
            # main = filo.iloc[i]['MAIN_ID']
            # if tic == 24693383: pdb.set_trace()

            if type(otype) == np.float: # >> will skip unmatched TICIDs
                otype = 'NONE'
            else: # -- simplify object type names ------------------------------
                otype = otype.replace('+', '|')
                otype_list = otype.split('|')
                otype_list_new = []
                for o in otype_list: # >> loop through object types
                    if len(o) > 0:
                        while o[-1] in uncertainty_flags: # >> rmv uncertainty flags
                            o = o[:-1]
                            if len(o) == 0: break
                        if '(' in o: # >> remove (B)
                            o = o[:o.index('(')]
                        if o in list(renamed.keys()):
                            o = renamed[o]
                        if o in remove_classes:
                            o = ''
                        otype_list_new.append(o)
                        
                otype = list(np.unique(otype_list_new))
                if '' in otype:
                    otype.pop(otype.index('')) 
                otype = '|'.join(otype)
                if otype == '':
                    otype = 'NONE'

                # if int(tic) == 243242576: pdb.set_trace()

            with open(out_f, 'a') as f:
                f.write(str(tic)+','+str(otype)+','+str(main)+'\n')
                # f.write(align%(tic, otype, main)+'\n')

        print('Wrote '+out_f)

def clean_asassn(metapath, uncertainty_flags=[':', '?', '*']):

    # -- classes to be removed -------------------------------------------------
    remove_classes = make_remove_class_list()

    # -- produce revised cross-matched classifications -------------------------

    fnames = [f for f in os.listdir(metapath+'spoc/cat/') if 'asassn_raw' in f]
    fnames.sort()

    for fname in fnames:
        filo = np.loadtxt(metapath+'spoc/cat/'+fname, delimiter=',',
                          dtype='str', skiprows=1)
        # filo = pd.read_csv(metapath+'spoc/cat/'+fname, delimiter='\s+,')
        # >> original fname ends with '_asassn_raw.txt'
        # >> output fname ends with '_asassn.txt'
        out_f = metapath+'spoc/cat/'+fname[:-8]+'_cln.txt' 
        with open(out_f, 'w') as f:
            f.write('TICID,TYPE,ASASSN_NAME\n')
            # f.write(align%('TICID','TYPE','ASASSN_NAME')+'\n')

        for i in range(len(filo)):
            tic = filo[i][0]
            otype = filo[i][1]
            main = filo[i][2]
            # tic = filo.iloc[i]['TICID']
            # otype = filo.iloc[i]['TYPE']
            # main = filo.iloc[i]['ASASSN_NAME']

            otype = otype.replace('+', '|')
            otype_list = otype.split('|')
            otype_list_new = []
            for o in otype_list: # >> loop through object types
                if len(o) > 0:
                    if o[-1] in uncertainty_flags: # >> rmv uncertainty flag
                        o = o[:-1]
                    if o in remove_classes:
                        o = ''
                    otype_list_new.append(o)
                        
                otype = list(np.unique(otype_list_new))
                if '' in otype:
                    otype.pop(otype.index('')) 
                otype = '|'.join(otype)
                if otype == '':
                    otype = 'NONE'

            with open(out_f, 'a') as f:
                f.write(str(tic)+','+str(otype)+','+str(main)+'\n')
                # f.write(align%(tic, otype, main)+'\n')

        print('Wrote '+out_f)


# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# :: Organizing object types :::::::::::::::::::::::::::::::::::::::::::::::::::
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

def get_otypes(catpath, string='_raw', output_dir=None, skiprows=1):
    fnames = [f for f in os.listdir(catpath) if string in f]
    fnames.sort()
    otypes = []
    for fname in fnames:
        filo = np.loadtxt(catpath+fname, delimiter=',',
                          dtype='str', skiprows=skiprows)
        otypes.extend(filo[:,1])
    print(np.unique(otypes))

    if type(output_dir) != type(None):
        var_d = get_var_descr()
        out_f = output_dir+'otypes'+string+'.txt'
        with open(out_f, 'w') as f:
            f.write('### Object type, description  ###\n')
            for otype in np.unique(otypes):
                if otype in list(var_d.keys()):
                    f.write(otype+','+var_d[otype]+'\n')
                else:
                    f.write(otype+',\n')        

        print('Wrote '+out_f)

        out_f = output_dir+'otypes_bar'+string+'.png'
        all_otypes = []
        for o in otypes:
            all_otypes.extend(o.split('|'))
        all_otypes, counts = np.unique(all_otypes, return_counts=True)
        inds = np.argsort(counts)
        all_otypes, counts = all_otypes[inds], counts[inds]
        plt.figure(figsize=(np.max([len(all_otypes)/3, 7]),10))
        plt.bar(all_otypes, counts)
        plt.yscale('log')
        plt.xticks(rotation='vertical')
        plt.xlabel('Object types')
        plt.ylabel('Counts')
        plt.tight_layout()
        plt.savefig(out_f)
        print('Wrote '+out_f)
        
        
def get_var_descr():
    '''Reads docs/var_descr.txt to create a dictionary, where keys are the
    variability type abbrievations, and the values are human-understandable
    descriptions.'''
    d = {}
    with open('tess_stellar_var/docs/var_descr.txt', 'r') as f:
        lines = f.readlines()
        for line in lines[1:]:
            key = line.split(',')[0]
            val = ','.join(line.split(',')[1:])[:-1]
            d[key] = val
    return d        


def make_parent_dict():    
    d = {'I': ['IA', 'IB'],
         'IN': ['FU', 'INA', 'INB', 'INTIT', 'IN(YY)', 'INAT', 'INS', 'INSA',
                'INSB', 'INST', 'INT', 'INT(YY)', 'IT'],
         'IS': ['ISA', 'ISB'],
         'Fl': ['UV', 'UVN'],
         
         'BCEP': ['BCEPS', 'BCEP(B)'],
         'CEP': ['CEP(B)', 'DCEP', 'DCEPS'],
         'CW': ['CWA', 'CWB'],
         'DSCT': ['DSCTC', 'DSCTC(B)'],
         'L': ['LB', 'LC', 'LPB', 'LP', 'LBV'],
         'RR': ['RR(B)', 'RRAB', 'RRC'],
         'RV': ['RVA', 'RVB'],
         'SR': ['SRA', 'SRB', 'SRC', 'SRD', 'SRS'],
         'ZZ': ['ZZA', 'ZZB', 'ZZO', 'ZZLep'],
         
         'ACV': ['ACVO'],
         
         'N': ['NA', 'NB', 'NC', 'NL', 'NR'],
         'SN': ['SNI', 'SNII'],
         'UG': ['UGSS', 'UGSU', 'UGZ'],
         
         # 'E': ['EA', 'EB', 'EP', 'EW'],
         'D': ['DM', 'DS', 'DW'],
         'K': ['KE', 'KW'],
         
         'X': ['XB', 'XF', 'XI', 'XJ', 'XND', 'XNG', 'XP', 'XPR',
               'XPRM', 'XM', 'XRM', 'XN','XNA','XNGP','XPM','XPNG',
               'XNP'],
         }    

    parents = list(d.keys())

    # >> turn into array
    subclasses = []
    for parent in parents:
        subclasses.extend(d[parent])

    return d, parents, subclasses

def make_variability_tree():
    # >> renamed 'INT,IT' type as 'INTIT' since I used ',' as txt delimiter
    var_d = {'eruptive':
         ['Fl', 'BE', 'FU', 'GCAS', 'I', 'IA', 'IB', 'IN', 'INA', 'INB', 'INTIT',
          'IN(YY)', 'IS', 'ISA', 'ISB', 'RCB', 'RS', 'SDOR', 'UV', 'UVN',
          'WR', 'GCAS', 'out', 'Ae', 'Er'],
         'pulsating':
             ['Pu', 'ACYG', 'BCEP', 'BCEPS', 'BLBOO', 'CEP', 'CEP(B)', 'CW', 'CWA',
              'CWB', 'DCEP', 'DCEPS', 'DSCT', 'DSCTC', 'GDOR', 'L', 'LB', 'LC',
              'LPB', 'M', 'PVTEL', 'RPHS', 'RR', 'RR(B)', 'RRAB', 'RRC', 'RV',
              'RVA', 'RVB', 'SR', 'SRA', 'SRB' 'SRC', 'SRD', 'SRS', 'SXPHE',
              'ZZ', 'ZZA', 'ZZB', 'ZZO', 'SX'],
         'rotating': ['ACV', 'ACVO', 'BY', 'ELL', 'FKCOM', 'PSR',
                      'R', 'SXARI', 'ROT', 'CROT'],
         'cataclysmic':
             ['N', 'NA', 'NB', 'NC', 'NL', 'NR', 'SN', 'SNI', 'SNII', 'UG',
              'UGSS', 'UGSU', 'UGZ', 'ZAND', 'DQ', 'CV'],
         'eclipsing':
             ['E', 'EA', 'EB', 'EP', 'EW', 'GS', 'PN', 'RS', 'WD', 'WR', 'AR',
              'D', 'DM', 'DS', 'DW', 'K', 'KE', 'KW', 'SD'],
             'xray':
             ['AM', 'X', 'XB', 'XF', 'XI', 'XJ', 'XND', 'XNG', 'XP', 'XPR', 'XPRM',
              'XM'],
             'other': ['VAR']} 
    return var_d

def make_redundant_otype_dict():
    # >> keys are redundant object types, and will be removed if star is also
    # >> classified as any of the associated dictionary values
    var_d = make_variability_tree()

    d = {'**': var_d['eclipsing']+['R'],
         'E':  ['EA', 'EB', 'EP', 'EW', 'GS', 'PN', 'RS', 'WD', 'WR', 'AR', 'D',
                'DM', 'DS', 'DW', 'K', 'KE', 'KW', 'SD'],
         'Er': var_d['eruptive'],
         'ROT': var_d['rotating'],
         'Ro': var_d['rotating'],
         'Pu': var_d['pulsating'],
         'L': var_d['eruptive']+var_d['rotating']+\
         var_d['cataclysmic']+var_d['eclipsing']+var_d['xray']+var_d['other']+\
        ['Pu', 'ACYG', 'BCEP', 'BCEPS', 'BLBOO', 'CEP', 'CEP(B)', 'CW', 'CWA',
              'CWB', 'DCEP', 'DCEPS', 'DSCT', 'DSCTC', 'GDOR', 'LB', 'LC',
              'LPB', 'M', 'PVTEL', 'RPHS', 'RR', 'RR(B)', 'RRAB', 'RRC', 'RV',
              'RVA', 'RVB', 'SR', 'SRA', 'SRB' 'SRC', 'SRD', 'SRS', 'SXPHE',
              'ZZ', 'ZZA', 'ZZB', 'ZZO'],
         'LP': var_d['eruptive']+var_d['pulsating']+var_d['rotating']+\
         var_d['cataclysmic']+var_d['eclipsing']+var_d['xray']+var_d['other'],
         'RR': ['CEP'],
         'X': var_d['xray'],
         'BE': ['GCAS'],
         'CV': var_d['cataclysmic'],
         'DSCT(B:)': ['DSCT']
         }
    parents = list(d.keys())

    # >> turn into array
    subclasses = []
    for parent in parents:
        subclasses.extend(d[parent])

    return d, parents, subclasses


def merge_otype(otype_list, debug=True):

    # >> merge classes
    parent_dict, parents, subclasses = make_parent_dict()

    new_otype_list = []
    for otype in otype_list:
        if type(otype) == np.float: # >> if nan
            otype = 'NONE'
        if otype[0] == '|': 
            otype[1:]
        if otype in subclasses:
            # >> find parent
            for parent in parents:
                if otype in parent_dict[parent]:
                    new_otype = parent
            new_otype_list.append(new_otype)
        else:
            new_otype_list.append(otype)
    otype_list = new_otype_list

    # >> remove redundant classes 
    redundant_dict, parents, subclasses = make_redundant_otype_dict()
    new_otype_list = []
    for otype in otype_list:
        if otype in parents:
            if len(np.intersect1d(redundant_dict[otype], otype_list))>0:
                new_otype_list.append('')
            else:
                new_otype_list.append(otype)
        else:
            new_otype_list.append(otype)    

    if debug: pdb.set_trace()
    otype_list = np.unique(new_otype_list).astype('str')
    otype_list = np.delete(otype_list, np.where(otype_list == ''))
    if len(new_otype_list) > 1:
        otype_list = np.delete(otype_list, np.where(otype_list == 'NONE'))
    return otype_list


def get_parent_otypes(ticid, otypes):
    '''Finds all the objects with same parent and combines them into the same
    class
    '''

    parent_dict = make_parent_dict()
    parents = list(parent_dict.keys())

    # >> turn into array
    subclasses = []
    for parent in parents:
        subclasses.extend(parent_dict[parent])

    new_otypes = []
    for i in range(len(otypes)):
        otype = otypes[i].split('|')

        new_otype=[]
        for o in otype:
            if o in subclasses:
                for parent in parents: # >> find parent otype
                    if o in parent_dict[parent]:
                        new_o = parent
                new_otype.append(new_o)
            else:
                new_otype.append(o)

        # >> remove repeats
        new_otype = np.unique(new_otype)

        # >> remove parent if child in otype list e.g. E|EA or E|EW is redundant
        for parent in parents:
            if parent in new_otype:
                if len(np.intersect1d(new_otype, parent_dict[parent]))>0:
                    new_otype = np.delete(new_otype,
                                          np.nonzero(new_otype==parent))

        new_otypes.append('|'.join(new_otype.astype('str')))
    
    # >> get rid of empty classes
    new_otypes = np.array(new_otypes)
    inds = np.nonzero(new_otypes == '')
    new_otypes = np.delete(new_otypes, inds)
    ticid = np.delete(ticid, inds)
                
    return ticid, new_otypes



def get_parents_only(class_info, parent_dict=None,
                     remove_flags=[]):
    '''Finds all the objects with same parent and combines them into the same
    class
    TODO: get rid of this function
    '''
    classes = []
    new_class_info = []

    if type(parent_dict) == type(None):
        parent_dict = make_parent_dict()

    parents = list(parent_dict.keys())

    # >> turn into array
    subclasses = []
    for parent in parents:
        subclasses.extend(parent_dict[parent])

    for i in range(len(class_info)):
        otype_list = class_info[i][1]

        # >> remove any flags
        for flag in remove_flags:
            otype_list = otype_list.replace(flag, '|')
        otype_list = otype_list.split('|')

        new_otype_list=[]
        for otype in otype_list:
            if otype in subclasses:
                # >> find parent
                for parent in parents:
                    if otype in parent_dict[parent]:
                        new_otype = parent

                new_otype_list.append(new_otype)
            else:
                new_otype_list.append(otype)

        # >> remove repeats
        new_otype_list = np.unique(new_otype_list)

        # >> don't want e.g. E|EA or E|EW (redundant)
        if 'E' in new_otype_list:
            if len(np.intersect1d(new_otype_list, ['EA', 'EP', 'EW', 'EB']))>0:
                new_otype_list = np.delete(new_otype_list, np.nonzero(new_otype_list=='E'))

        if 'L' in new_otype_list and len(new_otype_list) > 1:
            new_otype_list = np.delete(new_otype_list,
                                       np.nonzero(new_otype_list=='L'))

        if '' in new_otype_list:
            new_otype_list = np.delete(new_otype_list,
                                       np.nonzero(new_otype_list==''))

        # if '|'.join(new_otype_list) == '|AR|EA|EB|RS|SB':
        #     pdb.set_trace()

        new_class_info.append([class_info[i][0], '|'.join(new_otype_list),
                               class_info[i][2]])

    # >> get rid of empty classes
    new_class_info = np.array(new_class_info)
    new_class_info = np.delete(new_class_info,
                               np.nonzero(new_class_info[:,1]==''), 0)
            
    
    return new_class_info

def make_remove_class_list(catalog='', rmv_flagged=True):
    '''Currently, our pipeline can only do clustering based on photometric data.
    So classes that require spectroscopic data, etc. are removed.'''
    rmv = ['SB', 'blu', 'EmO',
           'S', 'Rad', 'C', 'G', 'Sy', 'Sy1', 'Sy2',
           '*', 'NIR', 'Ir', 'smm', 'mR', 'cm', 'mm', 'HI', 'rB', 'Mas',
           'FIR', 'MIR', 'NIR', 'UX', 'ULX', 'IR', 'gB', 'S', 'HII',
           'UVem', 'Xem', 'Em', 'gam']
    sequence_descriptors = ['AB', 'HS', 'BS', 'YSO', 'Y', 'sg', 'BD', 's*b',
                            'Y*', 's?r', 's?b', 's*y', 'Y*O', 'RG', 'HB',
                            's*r', 'WD', 'pA', 'LM', 'OH', 's?y', 'cor', 'bluOB',
                            'As', 'LM', 'HB']
    # non_star = ['AGN', 'PoC', 'RNe', 'RB', 'OpC', 'LSB', 'ISM', 'GiP', 'GiC' , 'EmG',
    #             'DNe', 'ClG', 'Bla', 'BH', 'QSO', 'Neu']

    non_star = ['PoC', 'RNe', 'RB', 'OpC', 'LSB', 'ISM', 'GiP', 'GiC' , 'EmG',
                'DNe', 'ClG', 'BH', 'Neu', 'LSB', 'RB']

    not_descriptive = ['ev', 'W', 'RRD', 'RO', 'Q', 'HX', 'HH', 'HADS', 'Bz', 'BL', 'B',
                       'PM', 'nan', 'V', 'VAR', 'HV', 'mul']

    if rmv_flagged:
        flagged = ['Pe', 'I', 'IA', 'IB', 'IS', 'ISA', 'ISB'] # make_flagged_class_list()
    else:
        flagged = []

    # return rmv+sequence_descriptors+flagged+non_star+not_descriptive
    return rmv+sequence_descriptors+not_descriptive+non_star

def make_flagged_eclip_list():
    # >> section 5bc of GCVS classifications: eclipsing systems classified
    # >> based on physical characteristics rather than shape of light curve
    eclip = ['GS', 'PN', 'RS', 'WD', 'WR', 'AR', 'D', 'DM', 'DS', 'DW', 'K',
             'KE', 'KW', 'SD'] 
    return eclip

# def make_remove_class_list(simbad=False, rmv_flagged=True):
#     '''Currently, our pipeline can only do clustering based on photometric data.
#     So classes that require spectroscopic data, etc. are removed.'''
#     rmv = ['PM', 'IR', 'nan', 'V', 'VAR', 'As', 'SB', 'LM', 'blu', 'EmO', 'S',
#            ]
#     sequence_descriptors = ['AB', 'HS', 'BS', 'YSO', 'Y', 'sg', 'BD', 's*b']

#     if simbad:
#         rmv.append('UV')
#         rmv.append('X')

#     if rmv_flagged:
#         flagged = ['Em', 'Pe', 'gam']
#     else:
#         flagged = []

#     return rmv+sequence_descriptors+flagged

def write_true_label_txt(metapath, rmv_flagged=True,
                         catalogs=['gcvs', 'simbad', 'asassn']):
    '''Combine cat/*_cln.txt, as well as obj/OTYPE.txt '''

    sectors = [f[:10] for f in os.listdir(metapath+'spoc/cat/')]
    sectors = np.unique(np.array(sectors))

    obj_names = [f[:-4] for f in os.listdir(metapath+'spoc/obj/')]
    obj_ids = []
    for i in range(len(obj_names)):
        obj_ids.append(np.loadtxt(metapath+'spoc/obj/'+obj_names[i]+'.txt',
                                  skiprows=1).astype('int'))
    
    for sector in sectors:
        s = int(sector[7:-1])
        ticid = np.loadtxt(metapath+'spoc/targ/2m/all_targets_S%03d'%s\
                           +'_v1.txt')[:,0].astype('int')
        # otypes = {key: np.empty(0) for key in ticid} # >> initialize
        otypes = {key: [] for key in ticid} # >> initialize

        for catalog in catalogs:
            filo = np.loadtxt(metapath+'spoc/cat/'+sector+catalog+'_cln.txt',
                              delimiter=',', dtype='str', skiprows=1)
            # filo = pd.read_csv(metapath+'spoc/cat/'+sector+catalog+'.txt',
            #                    delimiter='\s+,')
            for i in range(len(filo)):
                tic = int(filo[i][0])
                otypes[tic].extend(filo[i][1].split('|'))
                for j in range(len(obj_names)):
                    if int(tic) in obj_ids[j]:
                        otypes[tic].append(obj_names[j])

        # >> save to text file
        out = metapath+'spoc/true/'+sector+'true_labels.txt'
        with open(out, 'w') as f:
            f.write('### Variability classifications for stars observed with'+\
                    ' 2-min cadence during '+str(sector)+' ###\n')
            f.write('TICID,TYPE\n')
            # f.write(align%('TICID','TYPE')+'\n')
            for i in range(len(ticid)):
                debug=False
                # if int(ticid[i]) == 105769104:
                #     debug=True
                # else:
                #     debug=False
                
                # >> merge classes
                otype = merge_otype(otypes[ticid[i]], debug=debug)
                otype = '|'.join(otype)
                f.write(str(ticid[i])+','+str(otype)+'\n')
                # f.write(align%(ticid[i],otype)+'\n')
        print('Wrote '+out)

def chan_2021_true_otypes():

    var_d = {}
    descr_d = {}

    # -- Chan et al. 2021 classes ----------------------------------------------

    # >> Active Galactic Nuclei-like (AGNL)
    var_d['AGNL'] = ['QSO', 'AGN', 'BLLAC', 'Bla']
    descr_d['AGNL'] = 'Active Galactic Nuclei-like'

    # >> Cepheid (CEP)
    var_d['CEP'] = ['CEP', 'CEP(B)', 'DCEP', 'DCEPS', 'DSCT', 'DSCTC', 'BCEP',
                    'BCEPS', 'CW', 'CWA', 'CWB', 'RV', 'RVA', 'RVB',
                    'BLBOO']

    # >> Eclipsing Binaries (EB)
    var_d['EB'] = ['E', 'EA', 'EB', 'EP', 'EW', 'GS', 'PN', 'RS', 'WD', 'WR',
                   'AR', 'D', 'DM', 'DS', 'DW', 'K', 'KE', 'KW', 'SD']
    descr_d['EB'] = 'Eclipsing Binaries'

    # >> Long-Period Variables (LPV)
    var_d['LP'] = ['LP']
    descr_d['LP'] = 'Long-Period Variables'

    # >> Mira variables (Mira)
    var_d['M'] = ['M']
    descr_d['M'] = 'Mira Variables'

    # >> other Pulsating Variables (Pul_oth)
    var_d['Pul_oth'] = ['Pu', 'ACYG', 'GDOR', 'L', 'LB', 'LC', 'LPB', 'PVTEL',
                        'RPHS', 'SR', 'SRA', 'SRB' 'SRC', 'SRD', 'SRS', 'SXPHE',
                        'ZZ', 'ZZA', 'ZZB', 'ZZO', 'SX']
    descr_d['Pul_oth'] = 'Other Pulsating Variables'

    # >> RR Lyrae (RR)
    var_d['RR'] = ['RR', 'RR(B)', 'RRAB', 'RRC']
    descr_d['RR'] = 'RR Lyrae'

    # >> Peculiar (Pec)
    var_d['Pec'] = ['Pe', 'I', 'IA', 'IB']
    descr_d['Pec'] = 'Peculiar'

    # -- GCVS classes ----------------------------------------------------------
    var_d['CROT'] = ['CROT']
    descr_d['CROT'] = 'Complex Rotators'

    var_d['ROT'] = ['ACV', 'ACVO', 'BY', 'ELL', 'FKCOM', 'PSR',
                      'R', 'SXARI', 'ROT']
    descr_d['CROT'] = 'Other rotators'

    var_d['Er'] = ['Fl', 'BE', 'FU', 'GCAS', 'IN', 'INA',
                   'INB', 'INTIT', 'IN(YY)', 'IS', 'ISA', 'ISB', 'RCB', 'RS',
                   'SDOR', 'UV', 'UVN', 'WR', 'GCAS', 'out', 'Ae', 'Er']
    descr_d['Er'] = 'Eruptive Variables'

    var_d['N'] = ['N', 'NA', 'NB', 'NC', 'NL', 'NR', 'SN', 'SNI', 'SNII', 'UG',
                  'UGSS', 'UGSU', 'UGZ', 'ZAND', 'DQ', 'CV', 'AM']
    descr_d['N'] = 'Nova-like variables'

    return var_d, descr_d

def get_chan_merged_otypes(metapath):

    var_d, descr_d = chan_2021_true_otypes()
    unmerged = []
    merged = list(var_d.keys())
    for i in range(len(merged)):
        unmerged.extend(var_d[merged[i]])

    catpath = metapath+'spoc/true/'
    fnames = [f for f in os.listdir(catpath)]
    fnames.sort()
    ticid = []
    otypes = []
    for fname in fnames:
        filo = np.loadtxt(catpath+fname, delimiter=',',
                          dtype='str', skiprows=2)
        ticid.extend(filo[:,0].astype('int'))
        otypes.extend(filo[:,1])

    new_ticid = []
    new_otypes = []
    unclassified = []
    with open(metapath+'spoc/otypes_S1_26.txt', 'w') as f:
        f.write('### Object types for S1-26 using classes defined by '+\
                ' Chan et al., 2021 and GCVS ###\n')
        f.write('TICID,OTYPE\n')
        for i in range(len(ticid)):
            if i % 500 == 0: print('TICID '+str(i)+' / '+str(len(ticid)))
            if ticid[i] not in new_ticid:
                new_ticid.append(ticid[i])
                new_otype = []
                for o in otypes[i].split('|'):
                    if o in list(unmerged):
                        for parent in merged:
                            if o in var_d[parent]:
                                new_otype.append(parent)
                    else:
                        new_otype.append('')
                        unclassified.append(o)
                new_otype = np.unique(new_otype)
                if len(new_otype) > 1 and '' in new_otype:
                    ind = np.nonzero(new_otype == '')[0]
                    new_otype = np.delete(new_otype, ind)
                if len(new_otype) == 1 and '' in new_otype:
                    new_otype = ['UNCLASSIFIED']

                new_otype = '|'.join(new_otype)
                f.write(str(ticid[i])+','+new_otype+'\n')

    print(np.unique(unclassified))

            
# def read_otype_txt(otypes, otype_txt, data_dir, catalog=None, add_chars=['+', '/'],
#                    uncertainty_flags=[':', '?', '*'], rmv_flagged=True):

#     '''
#     Args:
#     * catalog: either NONE, 'simbad', 'asassn'
#     '''
#     rmv_classes = make_remove_class_list(catalog=catalog, rmv_flagged=rmv_flagged)

#     otype_dict = {}
#     if type(catalog) != type(None) and catalog != 'gcvs':
#         with open('./docs/var_'+catalog+'.txt', 'r') as f:
#             lines = f.readlines()
#         otype_dict = {}
#         for line in lines[1:]:
#             otype, otype_gcvs = line.split(',')
#             otype_gcvs = otype_gcvs.replace('\n', '')
#             otype_dict[otype] = otype_gcvs

#     with open(otype_txt, 'r') as f:
#         lines = f.readlines()
#         for i in range(len(lines)):
#             tic, otype, main_id = lines[i].split(',')

#             # >> make list of labels
#             for char in add_chars:
#                 otype = otype.replace(char, '|')
#             otype_list = otype.split('|')

#             # >> remove unceratinty flags
#             otype_list_new = []
#             stop=False
#             for o in otype_list:
#                 if o == 'UGSU':
#                     stop=True
#                 if len(o) > 0 and o != '**':
#                     # >> remove stars that aren't classified
#                     if o[0] == 'V':
#                         o = ''

#                     else:
#                         # >> remove uncertainty flags
#                         if o[-1] in uncertainty_flags:
#                             o = o[:-1]

#                         # >> convert to GCVS nomenclature
#                         if o in list(otype_dict.keys()): 
#                             o = otype_dict[o]

#                         # >> remove classes that require external information
#                         if o in rmv_classes:
#                             o = ''
#                     otype_list_new.append(o)

#             otypes[float(tic)] = np.append(otypes[float(tic)], np.unique(otype_list_new))

#     return otypes



def quick_simbad(ticidasstring):
    """ only returns if it has a tyc id"""
    catalogdata = Catalogs.query_object(ticidasstring, radius=0.02, catalog="TIC")[0]
    try: 
        tyc = "TYC " + catalogdata["TYC"]
        customSimbad = Simbad()
        customSimbad.add_votable_fields("otypes")
        res = customSimbad.query_object(tyc)
        objecttype = res['OTYPES'][0].decode('utf-8')
    except: 
        objecttype = "there is no TYC for this object"
    return objecttype



def get_true_var_types(ticid=[], data_dir='./', sector='all'):
    '''Reads Sector*_true_labels.txt, generated from make_true_label_txt()
    * 
    * sector: either 'all' or int'''
    ticid_true = []
    otypes = []
    database_dir = data_dir+'databases/'
    
    # >> find all text files in directory
    if sector == 'all':
        fnames = fm.filter(os.listdir(database_dir), '*_true_labels.txt')
    else:
        fnames = ['Sector'+str(sector)+'_true_labels.txt']
    
    for fname in fnames:
        data = np.loadtxt(database_dir+fname, delimiter=',', dtype='str',
                          skiprows=2)
        ticid_true.extend(data[:,0].astype('float'))
        otypes.extend(data[:,1])

    # >> only return classified targets in ticid list, if given
    if len(ticid) > 0:
        _, inds, _ = np.intersect1d(ticid_true, ticid, return_indices=True)
        ticid_true = np.array(ticid_true)[inds]
        otypes = np.array(otypes)[inds]
    
    return np.array(ticid_true), np.array(otypes)










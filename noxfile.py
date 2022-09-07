import nox

@nox.session
def tests(session):
    session.install('-r','tests/requirements.txt')
    session.install('.')
    session.run('pytest','--html=build/htmldoc/tests/report.html','--self-contained-html','--cov-config=coverage.cfg','--cov','AlgebraicNumber','tests','--doctest-modules','AlgebraicNumber','--junitxml=build/junit.xml')

@nox.session
def coverage(session):
    session.install('-r','tests/requirements.txt')
    session.run('coverage','html','-d','build/htmldoc/coverage')
    
@nox.session
def docs(session):
    session.install('-r','docs/requirements.txt')
    session.install('.')
    session.run('pdoc3','--config','latex_math=True','--force','--html','AlgebraicNumber','-o','build/htmldoc')
    session.run('pyreverse','-s0','AlgebraicNumber','-k','--colorized','-p','AlgebraicNumber','-m','no','-d','build/htmldoc')
    session.run('dot','-Tpng','build/htmldoc/classes_AlgebraicNumber.dot','-o','build/htmldoc/AlgebraicNumber/classes.png')

@nox.session
def build(session):
    session.install('wheel','twine','docutils')
    session.run('python','setup.py','sdist','bdist_wheel')
    